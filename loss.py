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
from utils import *

import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

opt = Options().parse()


def labeled_mut_loss(logpx_yz, mu_qz_xy, ls_qz_xy, y, logqy_x):
    L = logpx_yz+log_standard_categorical(y)+kld_latent_theano(mu_qz_xy, ls_qz_xy)
    classification_loss = - (y * logqy_x).sum(-1)
    return (L + classification_loss).mean()

def unlabelled_mut_loss(logpx_yz, mu_qz_xy, ls_qz_xy, qy_x, logqy_x, y):
    logpx_yz = logpx_yz.view(-1, y.shape[-1])
    py = log_standard_categorical(y).view(-1, y.shape[-1])
    kld = kld_latent_theano(mu_qz_xy, ls_qz_xy).view(-1, y.shape[-1])
    L = logpx_yz+py+kld
    exp_y = (torch.mul(qy_x, L) - torch.mul(qy_x, logqy_x)).sum(-1)
    return exp_y.mean()

def kld_latent_theano(mu, log_sigma):
    KLD_latent = 0.5 * (1.0 + 2.0 * log_sigma - mu ** 2.0 - (2.0 * log_sigma).exp()).sum(1)
    return KLD_latent

def classification_loss(y, logqy_x):
    return (y * logqy_x).sum(-1).mean()




def KLD_diag_gaussians_theano(mu, log_sigma, prior_mu, prior_log_sigma):
        """ KL divergence between two Diagonal Gaussians """
        # return prior_log_sigma - log_sigma + 0.5 * ((2. * log_sigma).exp() + (mu - prior_mu).sqrt()) * math.exp(-2. * prior_log_sigma) - 0.5
        return prior_log_sigma - log_sigma + 0.5 * ((2. * log_sigma).exp() + (mu - prior_mu)**2) * math.exp(-2. * prior_log_sigma) - 0.5

def sparse_theano(mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l):
    # print("sparse")
    # print(KLD_diag_gaussians_theano(mu_W1, logsigma_W1, 0.0, 0.0).sum())
    # print(KLD_diag_gaussians_theano(mu_S, logsigma_S, opt.mu_sparse, opt.logsigma_sparse).sum())
    return - (KLD_diag_gaussians_theano(mu_W1, logsigma_W1, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_b1, logsigma_b1, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_W2, logsigma_W2, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_b2, logsigma_b2, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_W3, logsigma_W3, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_b3, logsigma_b3, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_C, logsigma_C, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_l, logsigma_l, 0.0, 0.0).sum() + KLD_diag_gaussians_theano(mu_S, logsigma_S, opt.mu_sparse, opt.logsigma_sparse).sum())


