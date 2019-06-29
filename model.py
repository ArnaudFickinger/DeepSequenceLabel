###
'''
April 2019
Code by: Arnaud Fickinger
'''
###

import torch.nn as nn
import torch.nn.functional as F
from utils import *
import torch
import math
# torch.manual_seed(42)
import pickle

from options import Options

opt = Options().parse()

class M2_VAE(nn.Module):
    def __init__(self, latent_dim, sequence_length, alphabet_size, enc_h1_dim, enc_h2_dim, dec_h1_dim, dec_h2_dim, nb_diseases, nb_features, dim_h1_clas, dim_h2_clas, mu_sparse=None, sigma_sparse=None, cw_inner_dimension=None, nb_patterns=None, hasTemperature = None, hasDictionary = None):
        super(M2_VAE, self).__init__()

        self.encoder = Encoder(latent_dim, sequence_length, alphabet_size, enc_h1_dim, enc_h2_dim, nb_diseases, nb_features)

        if opt.stochastic_weigths:
            self.decoder = StochasticDecoder(latent_dim, sequence_length, alphabet_size, dec_h1_dim, dec_h2_dim, mu_sparse, sigma_sparse, cw_inner_dimension, nb_patterns, hasTemperature, hasDictionary, nb_diseases, nb_features)

        else:
            self.decoder = DeterministicDecoder(latent_dim, sequence_length, alphabet_size, dec_h1_dim, dec_h2_dim, mu_sparse, sigma_sparse, cw_inner_dimension, nb_patterns, hasTemperature, hasDictionary, nb_diseases, nb_features)

        self.classifier = Classifier(sequence_length, nb_features, dim_h1_clas, dim_h2_clas)

    def forward(self, x, y = None):
        is_labelled = False if y is None else True
        if not is_labelled:
            y = create_labels(x, opt.nb_label)
            x_repeated = x.repeat(opt.nb_label, 1)
        else:
            x_repeated = x
        mu, logsigma = self.encoder(torch.cat((x_repeated,y), 1))
        qy_x, logqy_x = self.classifier(x)
        z = sample_diag_gaussian_original(mu, logsigma)
        px_zy, logpx_zy, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l = self.decoder(torch.cat((z,y),1), x)
        return mu, logsigma, px_zy, logpx_zy, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l, qy_x, logqy_x

class DeterministicDecoder(nn.Module):
    def __init__(self, latent_dim, sequence_length, alphabet_size, h1_dim, h2_dim, mu_sparse, sigma_sparse,
                 cw_inner_dimension, nb_patterns, hasTemperature, hasDictionary, nb_diseases, nb_features):
        super(DeterministicDecoder, self).__init__()
        self.alphabet_size = alphabet_size
        self.sequence_length = sequence_length
        self.nb_patterns = nb_patterns
        self.mu_sparse = mu_sparse
        self.h2_dim = h2_dim
        self.logsigma_sparse = math.log(sigma_sparse)
        self.W1 = nn.Parameter(torch.Tensor(latent_dim + nb_diseases*nb_features, h1_dim).normal_(0, 0.01))
        self.b1 = nn.Parameter(
            torch.Tensor(h1_dim).normal_(0, 0.01))
        self.W2 = nn.Parameter(torch.Tensor(h1_dim, h2_dim).normal_(0, 0.01))
        self.b2 = nn.Parameter(
            torch.Tensor(h2_dim).normal_(0, 0.01))
        self.W3 = nn.Parameter(torch.Tensor(h2_dim, sequence_length * cw_inner_dimension).normal_(0,
                                                                                                     0.01))
        self.b3 = nn.Parameter(torch.Tensor(alphabet_size * sequence_length).normal_(0,
                                                                                        0.01))
        self.S = nn.Parameter(torch.Tensor(int(h2_dim / nb_patterns), sequence_length).normal_(0,
                                                                                                  0.01))
        self.C = nn.Parameter(torch.Tensor(cw_inner_dimension, alphabet_size).normal_(0,
                                                                                         0.01))
        self.l = nn.Parameter(torch.Tensor(1).normal_(0, 0.01))
        self.hasTemperature = hasTemperature
        self.hasDictionary = hasDictionary
        if opt.dropout>0:
            print("dropout")
            self.dropout = nn.Dropout(opt.dropout)

    def forward(self, z, x):

        S = self.S.repeat(self.nb_patterns, 1)  # W-scale
        S = F.sigmoid(S)
        if self.hasDictionary:
            W3 = self.W3.view(self.h2_dim * self.sequence_length, -1)
            W_out = torch.mm(W3, self.C)
            S = torch.unsqueeze(S, 2)
            W_out = W_out.view(-1, self.sequence_length, self.alphabet_size)
            W_out = W_out * S
            W_out = W_out.view(-1, self.sequence_length * self.alphabet_size)
        h1 = F.relu(F.linear(z, self.W1.t(), self.b1))  # todo print h1 with deterministic z
        if opt.dropout>0:
            h1 = self.dropout(h1)
        h2 = F.sigmoid(F.linear(h1, self.W2.t(), self.b2))
        if opt.dropout>0:
            h2 = self.dropout(h2)
        h3 = F.linear(h2, W_out.t(), self.b3)
        l = torch.log(1 + self.l.exp())
        h3 = h3 * l
        h3 = h3.view((-1, self.sequence_length, self.alphabet_size))
        px_zy = F.softmax(h3, 2)
        x = x.view(-1, self.sequence_length, self.alphabet_size)
        logpx_zy = (x * F.log_softmax(h3, 2)).sum(-1).sum(-1) #one-hot
        return px_zy, logpx_zy

class StochasticDecoder(nn.Module):
    def __init__(self, latent_dim, sequence_length, alphabet_size, h1_dim, h2_dim, mu_sparse, sigma_sparse,
                 cw_inner_dimension, nb_patterns, hasTemperature, hasDictionary, nb_diseases, nb_features):
        super(StochasticDecoder, self).__init__()
        self.alphabet_size = alphabet_size
        self.sequence_length = sequence_length
        self.nb_patterns = nb_patterns
        self.mu_sparse = mu_sparse
        self.h2_dim = h2_dim
        self.logsigma_sparse = math.log(sigma_sparse)
        self.mu_W1 = nn.Parameter(torch.Tensor(latent_dim + nb_diseases*nb_features, h1_dim).normal_(0, 0.01))
        self.logsigma_W1 = nn.Parameter(torch.Tensor(latent_dim + nb_diseases*nb_features, h1_dim).normal_(0, 0.01))
        self.mu_b1 = nn.Parameter(
            torch.Tensor(h1_dim).normal_(0, 0.01))  # try torch.randn(x,x,x)*0.01 if not working
        self.logsigma_b1 = nn.Parameter(torch.Tensor(h1_dim).normal_(0, 0.01))
        self.mu_W2 = nn.Parameter(torch.Tensor(h1_dim, h2_dim).normal_(0, 0.01))
        self.logsigma_W2 = nn.Parameter(torch.Tensor(h1_dim, h2_dim).normal_(0, 0.01))
        self.mu_b2 = nn.Parameter(
            torch.Tensor(h2_dim).normal_(0, 0.01))  # try torch.randn(x,x,x)*0.01 if not working
        self.logsigma_b2 = nn.Parameter(torch.Tensor(h2_dim).normal_(0, 0.01))
        self.mu_W3 = nn.Parameter(torch.Tensor(h2_dim, sequence_length * cw_inner_dimension).normal_(0,
                                                                                                     0.01))  # try torch.randn(x,x,x)*0.01 if not working
        self.logsigma_W3 = nn.Parameter(torch.Tensor(h2_dim, sequence_length * cw_inner_dimension).normal_(0, 0.01))
        self.mu_b3 = nn.Parameter(torch.Tensor(alphabet_size * sequence_length).normal_(0,
                                                                                        0.01))  # try torch.randn(x,x,x)*0.01 if not working
        self.logsigma_b3 = nn.Parameter(torch.Tensor(alphabet_size * sequence_length).normal_(0, 0.01))
        self.mu_S = nn.Parameter(torch.Tensor(int(h2_dim / nb_patterns), sequence_length).normal_(0,
                                                                                                  0.01))  # try torch.randn(x,x,x)*0.01 if not working
        self.logsigma_S = nn.Parameter(torch.Tensor(int(h2_dim / nb_patterns), sequence_length).normal_(0, 0.01))
        self.mu_C = nn.Parameter(torch.Tensor(cw_inner_dimension, alphabet_size).normal_(0,
                                                                                         0.01))  # try torch.randn(x,x,x)*0.01 if not working
        self.logsigma_C = nn.Parameter(torch.Tensor(cw_inner_dimension, alphabet_size).normal_(0, 0.01))
        self.mu_l = nn.Parameter(torch.Tensor(1).normal_(0, 0.01))
        self.logsigma_l = nn.Parameter(torch.Tensor(1).normal_(0, 0.01))
        self.hasTemperature = hasTemperature
        self.hasDictionary = hasDictionary
        if opt.dropout>0:
            print("dropout")
            self.dropout = nn.Dropout(opt.dropout)

    def forward(self, z, x):
        W1 = sample_diag_gaussian_original(self.mu_W1, self.logsigma_W1)
        b1 = sample_diag_gaussian_original(self.mu_b1, self.logsigma_b1)
        W2 = sample_diag_gaussian_original(self.mu_W2, self.logsigma_W2)
        b2 = sample_diag_gaussian_original(self.mu_b2, self.logsigma_b2)
        W3 = sample_diag_gaussian_original(self.mu_W3, self.logsigma_W3)
        b3 = sample_diag_gaussian_original(self.mu_b3, self.logsigma_b3)
        S = sample_diag_gaussian_original(self.mu_S, self.logsigma_S)
        S = S.repeat(self.nb_patterns, 1)  # W-scale
        S = F.sigmoid(S)
        if self.hasDictionary:
            W3 = W3.view(self.h2_dim * self.sequence_length, -1)
            C = sample_diag_gaussian_original(self.mu_C, self.logsigma_C)
            W_out = torch.mm(W3, C)
            S = torch.unsqueeze(S, 2)
            W_out = W_out.view(-1, self.sequence_length, self.alphabet_size)
            W_out = W_out * S
            W_out = W_out.view(-1, self.sequence_length * self.alphabet_size)
        h1 = F.relu(F.linear(z, W1.t(), b1))  # todo print h1 with deterministic z
        if opt.dropout>0:
            h1 = self.dropout(h1)
        h2 = F.sigmoid(F.linear(h1, W2.t(), b2))
        if opt.dropout>0:
            h2 = self.dropout(h2)
        h3 = F.linear(h2, W_out.t(), b3)
        l = sample_diag_gaussian_original(self.mu_l, self.logsigma_l)
        l = torch.log(1 + l.exp())
        h3 = h3 * l
        h3 = h3.view((-1, self.sequence_length, self.alphabet_size))
        px_z = F.softmax(h3, 2)
        x = x.view(-1, self.sequence_length, self.alphabet_size)
        logpx_z = (x * F.log_softmax(h3, 2)).sum(-1).sum(-1) #one-hot
        return px_z, logpx_z, self.mu_W1, self.logsigma_W1, self.mu_b1, self.logsigma_b1, self.mu_W2, \
                   self.logsigma_W2, self.mu_b2, self.logsigma_b2, self.mu_W3, self.logsigma_W3, self.mu_b3, \
                   self.logsigma_b3, self.mu_S, self.logsigma_S, self.mu_C, self.logsigma_C, self.mu_l, self.logsigma_l


class Encoder(nn.Module):
    def __init__(self, latent_dim, sequence_length, alphabet_size, h1_dim, h2_dim, nb_diseases, nb_features):
        super(Encoder, self).__init__()
        self.sequence_length = sequence_length
        self.alphabet_size = alphabet_size
        self.fc1 = nn.Linear(sequence_length*alphabet_size + nb_diseases*nb_features, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3_mu = nn.Linear(h2_dim, latent_dim)
        self.fc3_logsigma = nn.Linear(h2_dim, latent_dim)
        if opt.dropout>0:
            self.dropout = nn.Dropout(opt.dropout)

    def forward(self, x):

        if x.shape[-1]!= self.sequence_length*self.alphabet_size:
            x = x.view(-1, self.sequence_length*self.alphabet_size)

        h1 = F.relu(self.fc1(x))
        if opt.dropout>0:
            h1 = self.dropout(h1)
        h2 = F.relu(self.fc2(h1))
        if opt.dropout>0:
            h2 = self.dropout(h2)
        mu = self.fc3_mu(h2)
        logsigma = self.fc3_logsigma(h2)
        return mu, logsigma

class MultiClassifier(nn.Module):
    def __init__(self, input, nb_diseases, nb_features, dim_h1, dim_h2):
        super(MultiClassifier, self).__init__()
        self.fc1 = nn.Linear(input, dim_h1)
        self.fc2 = nn.Linear(dim_h1, dim_h2)
        self.fc3 = nn.Linear(dim_h2, nb_diseases*nb_features)
        self.nb_diseases = nb_diseases
        self.nb_features = nb_features

    def forward(self, x, y):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = self.fc3(h2)
        h3 = h3.reshape(-1, self.nb_diseases, self.nb_features)
        qy_z  = F.softmax(h3, 2)
        logqy_z = (y * F.log_softmax(h3,2))
        return qy_z

class Classifier(nn.Module):
    def __init__(self, input, nb_features, dim_h1, dim_h2):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input, dim_h1)
        self.fc2 = nn.Linear(dim_h1, dim_h2)
        self.fc3 = nn.Linear(dim_h2, nb_features)
        self.nb_features = nb_features

    def forward(self, x):
        x = x.shape(x.shape[0], -1)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = self.fc3(h2)
        qy_x  = F.softmax(h3, -1)
        logqy_x = F.log_softmax(h3,-1).sum(-1)
        return qy_x, logqy_x