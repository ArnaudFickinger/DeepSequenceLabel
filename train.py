###
'''
April 2019
Code by: Arnaud Fickinger
'''
###


from torch.utils.data import DataLoader
from dataset import *
from loss import *
from model import *
# from model_svae import *

import pandas as pd

from scipy.stats import spearmanr

import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from options import Options

opt = Options().parse()

str_vae = "svae"

def main():

    unlabeled_mutations, labeled_mutations, labels, healthy_seq = get_label(opt.gene)
    seqlen = len(healthy_seq)
    nb_features = labels.shape[1]
    nb_disease = 1
    alphabet_size = 20

    model = M2_VAE(opt.latent_dim, seqlen,
                              alphabet_size, opt.enc_h1_dim, opt.enc_h2_dim, opt.dec_h1_dim, opt.dec_h2_dim, nb_disease, nb_features, opt.dim_h1_clas, opt.dim_h2_clas, opt.mu_sparse, opt.logsigma_sparse, opt.cw_inner_dimension, opt.nb_patterns,
                              opt.has_temperature, opt.has_dictionary).to(device)

    u_seq = get_seq(unlabeled_mutations, healthy_seq)
    l_seq = get_seq(unlabeled_mutations, healthy_seq)

    u_one_hot = get_one_hot(u_seq)
    l_one_hot = get_one_hot(l_seq)

    labeled_dataset = Dataset_labelled(l_one_hot, labels)
    unlabeled_dataset = Dataset_unlabelled(u_one_hot)

    labeled_loader = DataLoader(labeled_dataset, batch_size=opt.batch_size)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=opt.batch_size)

    optimizer = torch.optim.Adam(model.parameters())

    u_loss = []
    clas_loss = []
    l_loss = []
    kld = []

    for l_batch, l_labels, u_batch in zip(labeled_loader, unlabeled_loader):
        u_batch = u_batch.to(device)
        l_batch = l_batch.to(device)
        l_labels = l_labels.to(device)
        optimizer.zero_grad()

        #labeled mutations
        mu_qz_xy, ls_qz_xy, px_zy, logpx_zy, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l, qy_x, logqy_x = model(l_batch, l_labels)
        loss_labeled = labeled_mut_loss(logpx_zy, mu_qz_xy, ls_qz_xy, l_labels, logqy_x)
        l_loss.append(loss_labeled.item())
        clas_loss.append(classification_loss(l_labels, logqy_x))
        kld_l = kld_latent_theano(mu_qz_xy, ls_qz_xy).mean().item()

        # unlabeled mutations
        mu_qz_xy, ls_qz_xy, px_zy, logpx_zy, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l, qy_x, logqy_x = model(u_batch)
        loss_unlabelled = unlabelled_mut_loss(logpx_zy, mu_qz_xy, ls_qz_xy, qy_x, logqy_x, l_labels)
        u_loss.append(loss_unlabelled.item())
        kld_u = kld_latent_theano(mu_qz_xy, ls_qz_xy).mean().item()
        kld.append(kld_l+kld_u)

        loss = loss_labeled + loss_unlabelled
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), opt.saving_path + "model_{}_{}_{}_{}_{}_{}.pth".format(str_vae, opt.latent_dim, opt.batch_size, -int(math.log10(opt.lr)), int(opt.neff), opt.epochs))

    titles = ["u_loss", "l_loss", "clas_loss", "kld"]

    plots = [u_loss, l_loss, clas_loss, kld]
    plt.clf()
    plt.figure()
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.plot(np.arange(len(plots[i])), plots[i])
        plt.title(titles[i])
    plt.suptitle(
        "ld: {}, bs: {}, lr: {}, e: {}".format(opt.latent_dim, opt.batch_size,
                                                              opt.lr, opt.epochs))
    plt.savefig(
        "plt_{}_{}_{}_{}_{}".format(str_vae, opt.latent_dim, opt.batch_size,
                                          -int(math.log10(opt.lr)), opt.epochs))
    plt.close('all')



if __name__ == "__main__":
    main()
