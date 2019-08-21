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

class VAE(nn.Module):
    def __init__(self, latent_dim, sequence_length, alphabet_size, enc_h1_dim, enc_h2_dim, dec_h1_dim, dec_h2_dim, dim_h1_clas, dim_h2_clas, mu_sparse=None, sigma_sparse=None, cw_inner_dimension=None, nb_patterns=None, hasTemperature = None, hasDictionary = None):
        super(VAE, self).__init__()

        self.encoder = Encoder(latent_dim, sequence_length*alphabet_size, enc_h1_dim, enc_h2_dim)

        if opt.stochastic_weigths:
            self.decoder = StochasticSparseDecoder(latent_dim, sequence_length, alphabet_size, dec_h1_dim, dec_h2_dim, mu_sparse, sigma_sparse, cw_inner_dimension, nb_patterns, hasTemperature, hasDictionary)

        else:
            self.decoder = DeterministicSparseDecoder(latent_dim, sequence_length, alphabet_size, dec_h1_dim, dec_h2_dim, mu_sparse, sigma_sparse, cw_inner_dimension, nb_patterns, hasTemperature, hasDictionary)


    def forward(self, x):
        x = x.float()
        x = x.view(x.shape[0], -1)
        mu, logsigma = self.encoder(x)
        z = sample_diag_gaussian_original(mu, logsigma)
        z = z.float()
        if opt.stochastic_weigths:
            px_z, logpx_z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l = self.decoder(z, x)
            return mu, logsigma, px_z, logpx_z, z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l
        else:
            px_z, logpx_z = self.decoder(z, x)
            return mu, logsigma, px_z, logpx_z, z

    def latent(self, x):
        mu, _ = self.encoder(x)
        return mu

class M2_VAE(nn.Module):
    def __init__(self, latent_dim, sequence_length, alphabet_size, enc_h1_dim, enc_h2_dim, dec_h1_dim, dec_h2_dim, nb_diseases, nb_features, dim_h1_clas, dim_h2_clas, mu_sparse=None, sigma_sparse=None, cw_inner_dimension=None, nb_patterns=None, hasTemperature = None, hasDictionary = None):
        super(M2_VAE, self).__init__()

        self.encoder = Encoder(latent_dim, sequence_length* alphabet_size, enc_h1_dim, enc_h2_dim, nb_diseases, nb_features)

        if opt.stochastic_weigths:
            self.decoder = StochasticSparseDecoder(latent_dim, sequence_length, alphabet_size, dec_h1_dim, dec_h2_dim, mu_sparse, sigma_sparse, cw_inner_dimension, nb_patterns, hasTemperature, hasDictionary, nb_diseases, nb_features)

        else:
            self.decoder = DeterministicSparseDecoder(latent_dim, sequence_length, alphabet_size, dec_h1_dim, dec_h2_dim, mu_sparse, sigma_sparse, cw_inner_dimension, nb_patterns, hasTemperature, hasDictionary, nb_diseases, nb_features)

        self.classifier = Classifier(sequence_length*alphabet_size, nb_features, dim_h1_clas, dim_h2_clas)
        self.nb_features = nb_features

    def forward(self, x, y = None):
        x = x.float()

        is_labelled = False if y is None else True
        x_flat = x.view(x.shape[0], -1)
        if not is_labelled:
            y = create_labels(x_flat, self.nb_features)
            x_repeated = x_flat.repeat(self.nb_features, 1)
        else:
            x_repeated = x_flat
        y = y.float()
        mu, logsigma = self.encoder(torch.cat((x_repeated,y), 1))
        qy_x, logqy_x = self.classifier(x)
        z = sample_diag_gaussian_original(mu, logsigma)
        z = z.float()
        if opt.stochastic_weigths:
            px_zy, logpx_zy, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l = self.decoder(torch.cat((z,y),1), x_repeated)
            return mu, logsigma, px_zy, logpx_zy,  qy_x, logqy_x, y, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l
        else:
            px_zy, logpx_zy = self.decoder(torch.cat((z,y),1), x_repeated)
            return mu, logsigma, px_zy, logpx_zy, qy_x, logqy_x, y

    def latent(self, x):
        x_flat = x.view(x.shape[0], -1)
        y = create_labels(x_flat, self.nb_features)
        x_repeated = x_flat.repeat(self.nb_features, 1)
        mu, _ = self.encoder(torch.cat((x_repeated, y), 1))
        mu = mu.view(-1, mu.shape[-1], y.shape[-1])
        qy_x, _ = self.classifier(x)
        qy_x = qy_x.unsqueeze(1)
        # print(x.shape)
        # print(qy_x.shape)
        # print(mu.shape)
        expec_y = (torch.mul(qy_x, mu)).sum(-1)
        return expec_y




class Stacked_M2_VAE(nn.Module):
    def __init__(self, latent_dim, input_dim, enc_h1_dim, enc_h2_dim, dec_h1_dim, dec_h2_dim, nb_diseases, nb_features, dim_h1_clas, dim_h2_clas, mu_sparse=None, sigma_sparse=None, cw_inner_dimension=None, nb_patterns=None, hasTemperature = None, hasDictionary = None):
        super(Stacked_M2_VAE, self).__init__()

        self.encoder = Encoder(latent_dim, input_dim, enc_h1_dim, enc_h2_dim, nb_diseases, nb_features)
        self.decoder = NormalDecoder(latent_dim+nb_diseases*nb_features, input_dim, dec_h1_dim, dec_h2_dim)
        self.classifier = Classifier(input_dim, nb_features, dim_h1_clas, dim_h2_clas)
        self.nb_features = nb_features

    def forward(self, z1, y = None):
        z1 = z1.float()

        is_labelled = False if y is None else True
        z1_flat = z1.view(z1.shape[0], -1)
        if not is_labelled:
            y = create_labels(z1_flat, self.nb_features)
            z1_repeated = z1_flat.repeat(self.nb_features, 1)
        else:
            z1_repeated = z1_flat
        y = y.float()
        mu, logsigma = self.encoder(torch.cat((z1_repeated,y), 1))
        qy_z1, logqy_z1 = self.classifier(z1)
        z2 = sample_diag_gaussian_original(mu, logsigma)
        z2 = z2.float()
        z1_r = self.decoder(torch.cat((z2, y), 1), z1_repeated)
        return mu, logsigma, z1_r, qy_z1, logqy_z1, y, z1_repeated

class M1_M2_VAE(nn.Module):
    def __init__(self, premodel, latent_dim_z1, latent_dim_z2, enc_h1_dim, enc_h2_dim, dec_h1_dim, dec_h2_dim, nb_diseases, nb_features, dim_h1_clas, dim_h2_clas, mu_sparse=None, sigma_sparse=None, cw_inner_dimension=None, nb_patterns=None, hasTemperature = None, hasDictionary = None):
        super(M1_M2_VAE, self).__init__()

        self.M1 = premodel
        self.M2 = Stacked_M2_VAE(latent_dim_z2, latent_dim_z1, enc_h1_dim, enc_h2_dim, dec_h1_dim, dec_h2_dim, nb_diseases, nb_features, dim_h1_clas, dim_h2_clas, mu_sparse=None, sigma_sparse=None, cw_inner_dimension=None, nb_patterns=None, hasTemperature = None, hasDictionary = None)

    def forward(self, x, y = None):

        _, _, _, _, z1 = self.M1(x)
        mu2, ls2, z1_r, qy_z1, logqy_z1, y, z1_repeated = self.M2(z1, y)

        return mu2, ls2, z1_r, qy_z1, logqy_z1, y, z1_repeated

class NormalDecoder(nn.Module): #for M1+M2 model, p(z1|z2) with normal assumption
    def __init__(self, latent_dim, input_dim, h1_dim, h2_dim):
        super(NormalDecoder, self).__init__()

        self.fc1 = nn.Linear(latent_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, input_dim)
        if opt.dropout > 0:
            self.dropout = nn.Dropout(opt.dropout)

    def forward(self, z, x):

        h1 = F.relu(self.fc1(z))
        h2 = F.relu(self.fc2(h1))
        x_r = self.fc3(h2)
        return x_r
        # h3 = h3.view((-1, self.sequence_length, self.alphabet_size))
        # p_z = F.softmax(h3, -1)
        # logpx_z = (x * F.log_softmax(h3, -1)).sum(-1)
        # return p_z, logpx_z

class DeterministicSparseDecoder(nn.Module):
    def __init__(self, latent_dim, sequence_length, alphabet_size, h1_dim, h2_dim, mu_sparse, sigma_sparse,
                 cw_inner_dimension, nb_patterns, hasTemperature, hasDictionary, nb_diseases = 0, nb_features = 0):
        super(DeterministicSparseDecoder, self).__init__()
        self.alphabet_size = alphabet_size
        self.sequence_length = sequence_length
        self.nb_patterns = nb_patterns
        self.mu_sparse = mu_sparse
        self.h2_dim = h2_dim
        self.logsigma_sparse = math.log(sigma_sparse)
        self.W1 = nn.Parameter(torch.Tensor(latent_dim + nb_diseases*nb_features, h1_dim))
        nn.init.xavier_normal_(self.W1)
        self.b1 = nn.Parameter(
            torch.Tensor(h1_dim))
        nn.init.constant_(self.b1, 0.1)
        self.W2 = nn.Parameter(torch.Tensor(h1_dim, h2_dim))
        nn.init.xavier_normal_(self.W2)
        self.b2 = nn.Parameter(
            torch.Tensor(h2_dim))
        nn.init.constant_(self.b2, 0.1)
        self.W3 = nn.Parameter(torch.Tensor(h2_dim, sequence_length * cw_inner_dimension))
        nn.init.xavier_normal_(self.W3)
        self.b3 = nn.Parameter(torch.Tensor(alphabet_size * sequence_length))
        nn.init.constant_(self.b3, 0.1)
        self.S = nn.Parameter(torch.Tensor(int(h2_dim / nb_patterns), sequence_length))
        nn.init.zeros_(self.mu_S)
        self.C = nn.Parameter(torch.Tensor(cw_inner_dimension, alphabet_size))
        nn.init.xavier_normal_(self.C)
        self.l = nn.Parameter(torch.Tensor(1))
        nn.init.ones_(self.mu_l)
        self.hasTemperature = hasTemperature
        self.hasDictionary = hasDictionary
        if opt.dropout>0:
            print("dropout")
            self.dropout = nn.Dropout(opt.dropout)

    def forward(self, z, x):

        x = x.float()

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
        # print(x.shape)
        # print(h3.shape)
        px_zy = F.softmax(h3, 2)
        x = x.view(-1, self.sequence_length, self.alphabet_size)
        # print(x.shape)
        # print(h3.shape)
        logpx_zy = (x * F.log_softmax(h3, 2)).sum(-1).sum(-1) #one-hot
        return px_zy, logpx_zy



class StochasticSparseDecoder(nn.Module):  # sparsity ideas of deep generative model for mutation paper
    def __init__(self, latent_dim, sequence_length, alphabet_size, h1_dim, h2_dim, mu_sparse, sigma_sparse,
                 cw_inner_dimension, nb_patterns, hasTemperature, hasDictionary, nb_diseases =0, nb_features = 0):
        super(StochasticSparseDecoder, self).__init__()

        self.alphabet_size = alphabet_size
        self.sequence_length = sequence_length
        self.nb_patterns = nb_patterns
        self.mu_sparse = mu_sparse
        self.h2_dim = h2_dim
        self.logsigma_sparse = math.log(sigma_sparse)
        self.mu_W1 = nn.Parameter(torch.Tensor(latent_dim+nb_features*nb_diseases, h1_dim))
        nn.init.xavier_normal_(self.mu_W1)
        self.logsigma_W1 = nn.Parameter(torch.Tensor(latent_dim+nb_features*nb_diseases, h1_dim))
        nn.init.constant_(self.logsigma_W1, -5)
        self.mu_b1 = nn.Parameter(
            torch.Tensor(h1_dim))
        nn.init.constant_(self.mu_b1, 0.1)
        self.logsigma_b1 = nn.Parameter(torch.Tensor(h1_dim))
        nn.init.constant_(self.logsigma_b1, -5)
        self.mu_W2 = nn.Parameter(torch.Tensor(h1_dim, h2_dim))
        nn.init.xavier_normal_(self.mu_W2)
        self.logsigma_W2 = nn.Parameter(torch.Tensor(h1_dim, h2_dim))
        nn.init.constant_(self.logsigma_W2, -5)
        self.mu_b2 = nn.Parameter(
            torch.Tensor(h2_dim))
        nn.init.constant_(self.mu_b2, 0.1)
        self.logsigma_b2 = nn.Parameter(torch.Tensor(h2_dim))
        nn.init.constant_(self.logsigma_b2, -5)
        self.mu_W3 = nn.Parameter(torch.Tensor(h2_dim, sequence_length * cw_inner_dimension))
        nn.init.xavier_normal_(self.mu_W3)
        self.logsigma_W3 = nn.Parameter(torch.Tensor(h2_dim, sequence_length * cw_inner_dimension))
        nn.init.constant_(self.logsigma_W3, -5)
        self.mu_b3 = nn.Parameter(torch.Tensor(alphabet_size * sequence_length))
        nn.init.constant_(self.mu_b3, 0.1)
        self.logsigma_b3 = nn.Parameter(torch.Tensor(alphabet_size * sequence_length))
        nn.init.constant_(self.logsigma_b3, -5)
        self.mu_S = nn.Parameter(torch.Tensor(int(h2_dim / nb_patterns), sequence_length))
        nn.init.zeros_(self.mu_S)
        self.logsigma_S = nn.Parameter(torch.Tensor(int(h2_dim / nb_patterns), sequence_length))
        nn.init.constant_(self.logsigma_S, -5)
        self.mu_C = nn.Parameter(torch.Tensor(cw_inner_dimension, alphabet_size))
        nn.init.xavier_normal_(self.mu_C)
        self.logsigma_C = nn.Parameter(torch.Tensor(cw_inner_dimension, alphabet_size))
        nn.init.constant_(self.logsigma_C, -5)
        self.mu_l = nn.Parameter(torch.Tensor(1))
        nn.init.ones_(self.mu_l)
        self.logsigma_l = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.logsigma_l, -5)
        self.hasTemperature = hasTemperature
        self.hasDictionary = hasDictionary
        if opt.dropout>0:
            print("dropout")
            self.dropout = nn.Dropout(opt.dropout)

    def forward(self, z, x):
        W1 = sample_diag_gaussian_original(self.mu_W1, self.logsigma_W1)  # "W_decode_"+str(layer_num)
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
        if False:
            h3 = h3.view((-1, opt.k_IWS, self.sequence_length, self.alphabet_size))
        else:
            h3 = h3.view((-1, self.sequence_length, self.alphabet_size))
        px_z = F.softmax(h3, -1)
        if False:
            x = x.view(-1, self.sequence_length, self.alphabet_size)
            x = x.unsqueeze(1)
            x = x.repeat(1, opt.k_IWS, 1, 1)
            x = x.view(-1, opt.k_IWS, self.sequence_length, self.alphabet_size)
        else:
            x = x.view(-1, self.sequence_length, self.alphabet_size)
        logpx_z = (x * F.log_softmax(h3, 2)).sum(-1).sum(-1)
        return px_z, logpx_z, self.mu_W1, self.logsigma_W1, self.mu_b1, self.logsigma_b1, self.mu_W2, \
                   self.logsigma_W2, self.mu_b2, self.logsigma_b2, self.mu_W3, self.logsigma_W3, self.mu_b3, \
                   self.logsigma_b3, self.mu_S, self.logsigma_S, self.mu_C, self.logsigma_C, self.mu_l, self.logsigma_l



class Encoder(nn.Module):
    def __init__(self, latent_dim, input_dim, h1_dim, h2_dim, nb_diseases = 0, nb_features = 0):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim + nb_diseases*nb_features, h1_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.1)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.1)
        self.fc3_mu = nn.Linear(h2_dim, latent_dim)
        nn.init.xavier_normal_(self.fc3_mu.weight)
        nn.init.constant_(self.fc3_mu.bias, 0.1)
        self.fc3_logsigma = nn.Linear(h2_dim, latent_dim)
        nn.init.xavier_normal_(self.fc3_logsigma.weight)
        nn.init.constant_(self.fc3_logsigma.bias, -5)
        if opt.dropout>0:
            self.dropout = nn.Dropout(opt.dropout)

    def forward(self, x):
        #
        x = x.view(x.shape[0], -1)



        x = x.float()

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
        # print(x.shape)
        # print("---")
        # print(x.shape)
        x = x.view(x.shape[0], -1).float()
        # print(x.shape)
        # print("---")
        # print(x.shape)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = self.fc3(h2)
        q_x  = F.softmax(h3, -1)
        logq_x = F.log_softmax(h3,-1)
        return q_x, logq_x