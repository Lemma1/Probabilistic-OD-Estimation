import os
import numpy as np
import pandas as pd
import hashlib
import time
import shutil
from scipy.sparse import coo_matrix
import multiprocessing as mp
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

import MNMAPI
from DODE import *
# from covariance_tree import *

def init(x):
	return torch.abs(nn.init.xavier_uniform_(x))


class torchDODE(nn.Module):
    def __init__(self, num_ODs, num_ints, mean_scale = 1.0, std_scale = 1.0):
        super(torchDODE, self).__init__()
        self.num_ODs = num_ODs
        self.num_ints = num_ints
        self.OD_flow_dims = num_ODs * num_ints
        self.mean_scale = mean_scale
        self.std_scale = std_scale
        self.initialize()

    def initialize(self, mean =None, std = None):
        if mean is None:
            self.log_od_mean = nn.Parameter(init(torch.Tensor(self.OD_flow_dims, 1))* self.mean_scale, requires_grad=True)
        else:
            self.log_od_mean = nn.Parameter(torch.from_numpy(mean), requires_grad=True)
        if std is None:
            self.log_od_std = nn.Parameter(init(torch.Tensor(self.OD_flow_dims, 1))* self.std_scale, requires_grad=True)
        else:
            self.log_od_std = nn.Parameter(torch.from_numpy(std), requires_grad=True)
        # print (self.log_od_mean, self.log_od_std)

    def generate_one_OD(self):
        q = self.reparameterize(self.log_od_mean, self.log_od_std)
        return q
        # tmp_sampled_q = torch.unsqueeze(self.reparameterize(tmp_logmean, tmp_logvar), 1)
        # # print (P, tmp_sampled_q)
        # tmp_f = torch.sparse.mm(P, tmp_sampled_q)
        # return tmp_f, tmp_sampled_q, tmp_logmean, tmp_logvar

    def compute_link(self, f, rho):
        # print (f)
        new_x = torch.sparse.mm(rho, f)
        return new_x

    def get_mu_std(self):
        mu = torch.exp(self.log_od_mean)
        # print (mu)
        std = torch.exp(self.log_od_std)
        return mu.detach().numpy().flatten(), std.detach().numpy().flatten()

    def reparameterize(self, logmu, logstd):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        mu = torch.exp(logmu)
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return eps * std + mu


class SDODE():
    def __init__(self, nb, config):
        self.config = config
        self.nb = nb
        self.num_assign_interval = nb.config.config_dict['DTA']['max_interval']
        self.ass_freq = nb.config.config_dict['DTA']['assign_frq']
        self.num_link = nb.config.config_dict['DTA']['num_of_link']
        self.num_path = nb.config.config_dict['FIXED']['num_path']
        self.num_loading_interval =  self.num_assign_interval * self.ass_freq
        self.data_dict = dict()
        self.num_data = self.config['num_data']
        self.observed_links = self.config['observed_links']
        self.all_links = list(map(lambda x: x.ID, self.nb.link_list))
        self.observed_links_idxs = list(map(lambda x: self.all_links.index(x), self.observed_links))
        self.paths_list = self.config['paths_list']
        self.demand_list = self.nb.demand.demand_list
        assert (len(self.paths_list) == self.num_path)

    def init_torch(self, init_mean_scale, init_std_scale):
        self.dode_solver = torchDODE(len(self.demand_list),
                        self.num_assign_interval, mean_scale = init_mean_scale,
                        std_scale = init_std_scale)

    def _add_link_flow_data(self, link_flow_df_list):
        assert (self.config['use_link_flow'])
        assert (self.num_data == len(link_flow_df_list))
        assert (len(self.observed_links) == len(link_flow_df_list[0].columns))
        for i in range(self.num_data):
            assert (len(self.observed_links) == len(link_flow_df_list[i].columns))
            for j in range(len(self.observed_links)):
                assert (self.observed_links[j] == link_flow_df_list[i].columns[j])
        self.data_dict['link_flow'] = link_flow_df_list

    def _add_link_tt_data(self, link_spd_df_list):
        assert (self.config['use_link_tt'])
        assert (self.num_data == len(link_spd_df_list))
        for i in range(self.num_data):
            assert (len(self.observed_links) == len(link_spd_df_list[i].columns))
            for j in range(len(self.observed_links)):
                assert (self.observed_links[j] == link_spd_df_list[i].columns[j])
        self.data_dict['link_tt'] = link_spd_df_list

    def add_data(self, data_dict):
        if self.config['use_link_flow']:
            self._add_link_flow_data(data_dict['link_flow'])
        if self.config['use_link_tt']:
            self._add_link_tt_data(data_dict['link_tt'])

    def _run_simulation(self, f, counter = 0):
        # print "RUN"
        hash1 = hashlib.sha1()
        hash1.update(str(time.time()) + str(counter))
        new_folder = str(hash1.hexdigest())
        self.nb.update_demand_path(f)
        self.nb.config.config_dict['DTA']['total_interval'] = self.num_loading_interval
        self.nb.dump_to_folder(new_folder)
        a = MNMAPI.dta_api()
        a.initialize(new_folder)
        shutil.rmtree(new_folder)
        a.register_links(self.all_links)
        a.register_paths(self.paths_list)
        a.install_cc()
        a.install_cc_tree()
        a.run_whole()
        return a

    def get_full_dar(self, dta, f):
        dar = dta.get_complete_dar_matrix(np.arange(0, self.num_loading_interval, self.ass_freq),
                np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq,
                self.num_assign_interval, f)
        return dar

    def get_path_tt_dict(self, dta):
        path_tt_array = dta.get_path_tt(np.arange(0, self.num_loading_interval, self.ass_freq))
        path_ID2tt = dict()
        for path_ID_idx, path_ID in enumerate(self.paths_list):
            path_ID2tt[path_ID] = path_tt_array[path_ID_idx, :]
        return path_ID2tt

    def assign_route_portions(self, path_ID2tt, theta = 0.1):
        for O_node in self.nb.path_table.path_dict.keys():
            for D_node in self.nb.path_table.path_dict[O_node].keys():
                tmp_path_set = self.nb.path_table.path_dict[O_node][D_node]
                cost_array = np.zeros((len(tmp_path_set.path_list),self.num_assign_interval))
                for tmp_path_idx, tmp_path in enumerate(tmp_path_set.path_list):
                    cost_array[tmp_path_idx, :] = path_ID2tt[tmp_path.path_ID]
                p_array = generate_portion_array(cost_array, theta = theta)
                for tmp_path_idx, tmp_path in enumerate(tmp_path_set.path_list):
                    tmp_path.attach_route_choice_portions(p_array[tmp_path_idx])

    # def compute_path_flow_grad_and_loss(self, one_data_dict, f, counter = 0):
    #     # print "Running simulation"
    #     dta = self._run_simulation(f, counter)
    #     # print "Getting DAR"
    #     dar = self.get_full_dar(dta, f)[self.get_full_observed_link_index(), :]
    #     # print "Evaluating grad"
    #     grad = np.zeros(len(self.observed_links) * self.num_assign_interval)
    #     if self.config['use_link_flow']:
    #         grad += self.config['link_flow_weight'] * self._compute_grad_on_link_flow(dta, one_data_dict)
    #     if self.config['use_link_tt']:
    #         grad += self.config['link_tt_weight'] * self._compute_grad_on_link_tt(dta, one_data_dict)
    #     # print "Getting Loss"
    #     loss = self._get_loss(one_data_dict, dta)
    #     new_path_cost_array = dta.get_path_tt(np.arange(0, self.num_loading_interval, self.ass_freq))
    #     return  dar.T.dot(grad), loss, new_path_cost_array
    #
    # def _compute_grad_on_link_flow(self, dta, one_data_dict):
    #     x_e = dta.get_link_inflow(np.arange(0, self.num_loading_interval, self.ass_freq),
    #               np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq).flatten(order = 'F')
    #     # print (x_e.shape, x_e[self.get_full_observed_link_index()].shape)
    #     # print (np.nan_to_num(one_data_dict['link_flow_mean'].shape))
    #     grad = -np.nan_to_num(link_flow_array - x_e[self.get_full_observed_link_index()])
    #     return grad

    # def _compute_grad_on_link_tt(self, dta, one_data_dict):
    #     tt_e = dta.get_link_tt(np.arange(0, self.num_loading_interval, self.ass_freq))
    #     tt_free = list(map(lambda x: self.nb.get_link(x).get_fft(), self.observed_links))
    #     for i in range(tt_e.shape[0]):
    #         pass
    #     return 0

    def _get_one_data(self, j):
        assert (self.num_data > j)
        one_data_dict = dict()
        if self.config['use_link_flow']:
            one_data_dict['link_flow'] = self.data_dict['link_flow'][j].values.flatten()
        if self.config['use_link_tt']:
            one_data_dict['link_tt'] = self.data_dict['link_tt'][j].values.flatten()
        return one_data_dict

    # def _get_loss(self, one_data_dict, dta):
    #     loss = np.float(0)
    #     if self.config['use_link_flow']:
    #         x_e = dta.get_link_inflow(np.arange(0, self.num_loading_interval, self.ass_freq),
    #               np.arange(0, self.num_loading_interval, self.ass_freq) + self.ass_freq).flatten(order = 'F')
    #         diff = np.nan_to_num(x_e[self.get_full_observed_link_index()] - one_data_dict['link_flow_mean'])
    #         loss += self.config['link_flow_weight'] * one_data_dict['link_flow_cov_inv'].dot(diff).dot(diff)
    #     return loss

    def get_full_observed_link_index(self):
        link_list = list()
        for i in range(self.num_assign_interval):
            for idx in self.observed_links_idxs:
                link_list.append(idx + i * len(self.all_links))
        return link_list

    def get_full_link_index(self):
        link_list = list(range(len(self.all_links) * self.num_assign_interval))
        return link_list

    def estimate_demand_cov(self, num_bucket = 5, init_mean_scale = 0.1,
                      init_std_scale = 0.01, step_size_mean = 0.1,
                      step_size_std = 0.1,
                      scheduler_step_size = 50, scheduler_gamma = 0.5,
                      max_epoch = 100, adagrad = False,
                      theta = 0.1, known_path_cost = None, save_folder = None,
                      true_dar = None, loss_name = 'l2'):
        from torch.optim.lr_scheduler import StepLR
        loss_list = list()
        self.init_torch(init_mean_scale, init_std_scale)
        # optimizer = torch.optim.Adam(self.dode_solver.parameters(), lr=step_size)
        optimizer_mean = torch.optim.Adadelta([self.dode_solver.log_od_mean], lr = step_size_mean)
        optimizer_std = torch.optim.Adadelta([self.dode_solver.log_od_std], lr = step_size_std)
        scheduler = StepLR(optimizer_mean, step_size=scheduler_step_size, gamma=scheduler_gamma)
        iter_counter = 0
        for i in range(max_epoch):
            scheduler.step()
            # print('Epoch:', i,'LR:', scheduler.get_lr())
            seq = np.random.permutation(self.num_data)
            seq_list = np.array_split(seq, num_bucket)
            loss = np.float(0)
            loss_emd = np.float(0)
            loss_kl = np.float(0)
            for one_seq in seq_list:
                ob_x_list = list()
                est_x_list = list()
                for j in one_seq:
                    # print ("one_seq", j)
                    # for e in self.dode_solver.parameters():
                    #     print(e)
                    one_data_dict = self._get_one_data(j)
                    true_x_numpy = one_data_dict['link_flow']
                    numpy_P = np.nan_to_num(self.nb.get_route_portion_matrix())
                    # print("numpy_P", numpy_P)
                    torch_P = convert_to_sparse_tensor(numpy_P)
                    # print("self.nb.get_route_portion_matrix()", self.nb.get_route_portion_matrix().toarray())
                    # print ("torch_P", torch_P)
                    torch_q = self.dode_solver.generate_one_OD()
                    # print('torch_f', torch_f)
                    # print (torch_q)
                    torch_f = torch.sparse.mm(torch_P, torch_q)
                    numpy_f = torch_f.detach().numpy().flatten()
                    # print("numpy_f", numpy_f)
                    tmp_dta = self._run_simulation(numpy_f, counter = iter_counter)
                    if true_dar is None:
                        numpy_dar = np.nan_to_num(self.get_full_dar(tmp_dta, numpy_f)[self.get_full_observed_link_index(), :])
                        # print("numpy_dar", numpy_dar.toarray())
                        torch_dar = convert_to_sparse_tensor(numpy_dar)
                    else:
                        torch_dar = convert_to_sparse_tensor(true_dar)
                    torch_x = self.dode_solver.compute_link(torch_f, torch_dar)
                    # print ("torch_f", torch_f)
                    # print ("torch_x", torch_x)
                    # print("torch_dar", torch_dar)
                    iter_counter += 1
                    ob_x_list.append(true_x_numpy)
                    est_x_list.append(torch_x)

                obs_mu, obs_cov = formulate_gaussian_from_numpy(ob_x_list)
                est_mu, est_cov = formulate_gaussian_from_torch(est_x_list)
                # print ('obs_cov',obs_cov)
                # print ('est_cov',est_cov)
                if loss_name == 'l2':
                    tmp_loss = approx_l2(obs_mu, obs_cov, est_mu, est_cov)
                elif loss_name == 'l1':
                    tmp_loss = approx_l1(obs_mu, obs_cov, est_mu, est_cov)
                elif loss_name == 'EMD':
                    tmp_loss = approx_EMD(obs_mu, obs_cov, est_mu, est_cov)
                elif loss_name == 'KL':
                    tmp_loss = approx_KL(obs_mu, obs_cov, est_mu, est_cov)
                elif loss_name == 'bhattacharyya':
                    tmp_loss = bhattacharyya(obs_mu, obs_cov, est_mu, est_cov)

                self.dode_solver.zero_grad()
                tmp_loss.backward()
                # print (self.dode_solver.get_mu_std())
                optimizer_mean.step()
                optimizer_std.step()
                loss += tmp_loss.detach().numpy()
                # loss_emd += tmp_loss1.detach().numpy()
                # loss_kl += tmp_loss2.detach().numpy()
                if known_path_cost is not None:
                    path_cost = known_path_cost
                else:
                    path_cost = tmp_dta.get_path_tt(np.arange(0, self.num_loading_interval, self.ass_freq))
                self.assign_route_portions(path_cost, theta = theta)
            # print (1, torch_x.t(), 2, torch.unsqueeze(tmp_link_flow,0))
            tmp_dict = generate_eval_metrics(obs_mu, obs_cov, est_mu, est_cov)
            print ("Epoch:", i, "Loss:", loss / np.float(self.num_data), tmp_dict)
            print (self.dode_solver.get_mu_std())
            loss_list.append((loss / np.float(self.num_data), tmp_dict))
            if save_folder is not None:
                pickle.dump([loss / np.float(self.num_data), self.dode_solver], open(os.path.join(save_folder, str(i)+'iteration.pickle'), 'wb'))
        return self.dode_solver, loss_list



### Distance related

# def exact_Wp(ob_x_list, est_x_list, p = 1, blur = 0.05):
#     from geomloss import SamplesLoss
#     obs_xs = torch.FloatTensor(np.array(x_list))
#     est_xs = torch.squeeze(torch.stack(x_list))
#     n_obs = obs_xs.shape[1]
#     n_est = est_xs.shape[1]
#     obs_ones = torch.ones(n_obs)
#     est_ones = torch.ones(n_est)

def generate_eval_metrics(obs_mu, obs_cov, est_mu, est_cov):
    tmp_dict = dict()
    tmp_dict['KL'] = np.float(approx_KL(obs_mu, obs_cov, est_mu, est_cov).detach().numpy())
    tmp_dict['EMD'] = np.float(approx_EMD(obs_mu, obs_cov, est_mu, est_cov).detach().numpy())
    tmp_dict['l2'] = np.float(approx_l2(obs_mu, obs_cov, est_mu, est_cov).detach().numpy())
    tmp_dict['l1'] = np.float(approx_l1(obs_mu, obs_cov, est_mu, est_cov).detach().numpy())
    tmp_dict['bhattacharyya'] = np.float(bhattacharyya(obs_mu, obs_cov, est_mu, est_cov).detach().numpy())
    return tmp_dict


def approx_KL(mu1, cov1, mu2, cov2):
    inv_cov2 = torch.pinverse(cov2)
    part1 = -torch.logdet(cov1)
    part2 = torch.trace(torch.mm(inv_cov2, cov1))
    mu_diff = torch.unsqueeze(mu2 - mu1, 1)
    # print (mu_diff)
    # print (torch.mm(inv_cov2, mu_diff))
    part3 = torch.sum(torch.mm(inv_cov2, mu_diff) * mu_diff)
    # print(part1, part2, part3)
    return part1 + part2 + part3


def approx_EMD(mu1, cov1, mu2, cov2):
    from sqrtm import sqrtm
    part1 = torch.sum(torch.pow(mu1 - mu2, 2))
    sqrt_cov2 = sqrtm(cov2)
    part2 = torch.trace(cov1 + cov2 - 2 * sqrtm(torch.mm(torch.mm(sqrt_cov2, cov1), sqrt_cov2)))
    # print (part1, part2)
    return  part1 + part2


def approx_l2(mu1, cov1, mu2, cov2):
    part1 = torch.sum(torch.pow(mu1 - mu2, 2))
    part2 = torch.sum(torch.pow(cov1 - cov2, 2))
    # print (part1, part2)
    return  part1 + part2


def approx_l1(mu1, cov1, mu2, cov2):
    part1 = torch.sum(torch.abs(mu1 - mu2))
    part2 = torch.sum(torch.abs(cov1 - cov2))
    # print (part1, part2)
    return  part1 + part2


def bhattacharyya(mu1, cov1, mu2, cov2):
    inv_cov2 = torch.pinverse(cov2)
    mu_diff = torch.unsqueeze(mu2 - mu1, 1)
    part1 = 0.125 * torch.sum(torch.mm(inv_cov2, mu_diff) * mu_diff)
    Sigma = (cov1+ cov2) / 2
    part2 = 0.5 * torch.logdet(Sigma) - 0.25 * torch.logdet(cov1) - 0.25 * torch.logdet(cov2 + eye_like(cov2) * 1e-12)
    # print (1111, torch.logdet(Sigma), torch.logdet(cov1), torch.logdet(cov2))
    return part1 + part2

def approx_emdapp(mu1, cov1, mu2, cov2):
    from sqrtm import sqrtm
    part1 = torch.sum(torch.pow(mu1 - mu2, 2))
    part2 = torch.sum(torch.pow(sqrtm(cov1) - sqrtm(cov2), 2))
    # print (part1, part2)
    return  part1 + part2


### Covariance related functions
def formulate_gaussian_from_numpy(x_list):
    xs = np.array(x_list)
    # print (xs.shape)
    mu = np.mean(xs, axis = 0)
    # print (mu)
    cov = np.cov(xs.T)
    # print (cov)
    cov = np.nan_to_num(cov)
    mu_torch = torch.FloatTensor(mu)
    cov_torch = torch.FloatTensor(cov)
    return mu_torch, cov_torch

def formulate_gaussian_from_torch(x_list, eps = 1e-6):
    xs = torch.squeeze(torch.stack(x_list))
    d = xs.shape[1]
    # print (xs)
    mu = torch.mean(xs, dim = 0)
    # cov = torch.cov(xs.t())
    cov = cov_torch(xs.t()) + torch.eye(d) * eps
    # print (mu, cov)
    return mu, cov

def cov_torch(X):
    D = X.shape[-1]
    mean = torch.mean(X, dim=-1).unsqueeze(-1).detach()
    # print (X)
    # print (mean)
    X = X - mean
    return torch.mm(X, X.t())/(D-1)

    # print (X)
    # return torch.mm(1/(D-1) * X, X.transpose(-1, -2))


def eye_like(tensor):
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor))


###  Behavior related function

def convert_to_sparse_tensor(M):
    """
    input: M is Scipy sparse matrix
    output: pytorch sparse tensor in GPU
    """
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    Ms = torch.sparse_coo_tensor(indices, values, shape)
    return Ms

def generate_portion_array(cost_array, theta = 0.1):
    p_array = np.zeros(cost_array.shape)
    for i in range(cost_array.shape[1]):
        p_array[:, i] = logit_fn(cost_array[:,i], theta)
    return p_array

def logit_fn(cost, theta, max_cut = False):
    scale_cost = - theta * cost
    if max_cut:
        e_x = np.exp(scale_cost - np.max(scale_cost))
    else:
        e_x = np.exp(scale_cost)
    return e_x / e_x.sum()
