###
'''
April 2019
Code by: Arnaud Fickinger
'''
###

import torch
import torch.nn.functional as F
import numpy as np
from options import Options
import math
# from utils import *

import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

opt = Options().parse()


def log_standard_categorical(p):

    prior = F.softmax(torch.ones_like(p), dim=1)
    prior.requires_grad = False

    cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)

    return cross_entropy

def labeled_mut_loss(logpx_yz, mu_qz_xy, ls_qz_xy, y, logqy_x):

    L = logpx_yz+log_standard_categorical(y)-kld_latent_theano(mu_qz_xy, ls_qz_xy)
    classification_loss = (torch.mul(y, logqy_x)).sum(-1)
    return (L + classification_loss).mean()

def unlabelled_mut_loss(logpx_yz, mu_qz_xy, ls_qz_xy, qy_x, logqy_x, y):
    logpx_yz = logpx_yz.view(-1, y.shape[-1])
    py = log_standard_categorical(y).view(-1, y.shape[-1])
    kld = kld_latent_theano(mu_qz_xy, ls_qz_xy).view(-1, y.shape[-1])
    L = logpx_yz+py-kld
    expec_y = (torch.mul(qy_x, L) - torch.mul(qy_x, logqy_x)).sum(-1)
    return expec_y.mean()

def unlabelled_mut_loss_no_mean(logpx_yz, mu_qz_xy, ls_qz_xy, qy_x, logqy_x, y):
    # print(logpx_yz.shape)
    logpx_yz = logpx_yz.view(-1, y.shape[-1])
    py = log_standard_categorical(y).view(-1, y.shape[-1])
    kld = kld_latent_theano(mu_qz_xy, ls_qz_xy).view(-1, y.shape[-1])
    L = logpx_yz+py-kld
    expec_y = (torch.mul(qy_x, L) - torch.mul(qy_x, logqy_x)).sum(-1)
    return expec_y

def labeled_mut_loss_normal(z1_rec, z1, mu_qz_xy, ls_qz_xy, y, logqy_x):
    # print("lab")
    nl = nll(z1, z1_rec)
    L = -nl+log_standard_categorical(y)-kld_latent_theano(mu_qz_xy, ls_qz_xy)
    classification_loss = (torch.mul(y, logqy_x)).sum(-1)
    return (L + classification_loss).mean()

def unlabelled_mut_loss_normal(z1_rec, z1, mu_qz_xy, ls_qz_xy, qy_x, logqy_x, y):
    # print("un")
    nl = nll(z1, z1_rec).view(-1, y.shape[-1])
    py = log_standard_categorical(y).view(-1, y.shape[-1])
    kld = kld_latent_theano(mu_qz_xy, ls_qz_xy).view(-1, y.shape[-1])
    # print(nl.shape)
    # print(py.shape)
    # print(kld.shape)
    L = -nl+py-kld
    expec_y = (torch.mul(qy_x, L) - torch.mul(qy_x, logqy_x)).sum(-1)
    return expec_y.mean()

def nll(x, xr):
    # print("x")
    # print(x.shape)
    # print(xr.shape)
    mse = ((xr - x) ** 2).view(x.shape[0], -1).mean(1)
    nll = mse / (2 * opt.var_rec)
    nll += 0.5 * math.log(opt.var_rec)
    return nll

def kld_latent_theano(mu, log_sigma):
    KLD_latent = -0.5 * (1.0 + 2.0 * log_sigma - mu ** 2.0 - (2.0 * log_sigma).exp()).sum(1)
    return KLD_latent

def classification_loss(y, logqy_x):
    return torch.mul(y, logqy_x).sum(-1).mean()

def loss_theano(mu, logsigma, z, px_z, logpx_z, Neff, warm_up_scale = 1.0, mu_W1 = None, logsigma_W1 = None, mu_b1 = None, logsigma_b1 = None, mu_W2 = None, logsigma_W2 = None, mu_b2 = None, logsigma_b2 = None, mu_W3 = None, logsigma_W3 = None, mu_b3 = None, logsigma_b3 = None, mu_S = None, logsigma_S = None, mu_C = None, logsigma_C = None, mu_l = None, logsigma_l = None):
    if mu_W1 is None:
        return (logpx_z-warm_up_scale*kld_latent_theano(mu, logsigma)).mean()
    else:
        return (logpx_z-warm_up_scale*kld_latent_theano(mu, logsigma)).mean()+warm_up_scale*sparse_theano(mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l)/Neff

def KLD_diag_gaussians_theano(mu, log_sigma, prior_mu, prior_log_sigma):
        """ KL divergence between two Diagonal Gaussians """
        # return prior_log_sigma - log_sigma + 0.5 * ((2. * log_sigma).exp() + (mu - prior_mu).sqrt()) * math.exp(-2. * prior_log_sigma) - 0.5
        return prior_log_sigma - log_sigma + 0.5 * ((2. * log_sigma).exp() + (mu - prior_mu)**2) * math.exp(-2. * prior_log_sigma) - 0.5

def sparse_theano(mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l):
    # print("sparse")
    # print(KLD_diag_gaussians_theano(mu_W1, logsigma_W1, 0.0, 0.0).sum())
    # print(KLD_diag_gaussians_theano(mu_S, logsigma_S, opt.mu_sparse, opt.logsigma_sparse).sum())
    return - (KLD_diag_gaussians_theano(mu_W1, logsigma_W1, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_b1, logsigma_b1, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_W2, logsigma_W2, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_b2, logsigma_b2, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_W3, logsigma_W3, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_b3, logsigma_b3, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_C, logsigma_C, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_l, logsigma_l, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_S, logsigma_S, opt.mu_sparse, opt.logsigma_sparse).sum())

def ELBO_no_mean(logpx_z, mu, logsigma, z, warm_up_scale):
    return logpx_z - warm_up_scale*kld_diag_gaussian_normal_original_no_mean(mu, logsigma)

def kld_diag_gaussian_normal_original_no_mean(mu, logsigma):
#     print(mu.shape)
#     print(type(mu))
    if len(mu.shape)<2:
        mu = mu.unsqueeze(1)
        logsigma = logsigma.unsqueeze(1)
#     if isScalar(mu):
#         print("scalar")
#         return 0.5 * (mu.pow(2) + torch.exp(2 * logsigma) - 2 * logsigma - 1)
    return 0.5 * (mu.pow(2) + (2 * logsigma).exp() - 2 * logsigma - 1).sum(1)
