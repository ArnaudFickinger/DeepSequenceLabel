###
'''
July 2019
Code by: Arnaud Fickinger
'''
###
import copy

import torch
from torch.utils.data import DataLoader
from dataset import *
from loss import *
from model import *
from itertools import cycle
# from model_svae import *

import scipy.stats as stats
import math
from sklearn.decomposition import PCA

from scipy.stats import spearmanr

import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

from options import Options

opt = Options().parse()

str_vae = "svae"

def create_joined_dataset(labeled_loader, unlabeled_loader):
    if len(labeled_loader)>len(unlabeled_loader):
        return zip(labeled_loader, cycle(unlabeled_loader))
    elif len(labeled_loader)<len(unlabeled_loader):
        return zip(cycle(labeled_loader), unlabeled_loader)
    else:
        return zip(labeled_loader, unlabeled_loader)

if opt.test_algo:
    nb_collection = 2
else:
    nb_collection= 10

def main():
    collections_dic = {}
    collections_dic['labelled_mut_dic']= []
    collections_dic['mut_pred_dic']= []
    collections_dic['latent_dic']= []
    collections_dic['stat_dic'] = []
    for collection in range(nb_collection):
        main_per_collection(collections_dic, collection)
    mut_pred_dic_mean = {experience:{set_:{path:{epoch:np.mean([dic[experience][set_][epoch] for dic in collections_dic['mut_pred_dic']],0) for epoch in collections_dic['mut_pred_dic'][experience][set_][path]}for path in collections_dic['mut_pred_dic'][experience][set_]} for set_ in collections_dic['mut_pred_dic'][experience]}for experience in collections_dic['mut_pred_dic']}
    latent_dic_mean = {experience:{set_:{path:{epoch:np.mean([dic[experience][set_][epoch] for dic in collections_dic['latent_dic']],0) for epoch in collections_dic['latent_dic'][experience][set_][path]} for path in collections_dic['mut_pred_dic'][experience][set_]}for set_ in collections_dic['latent_dic'][experience]}for experience in collections_dic['latent_dic']}
    stat_dic_mean = {experience:{set_:{type:np.mean([dic[experience][set_][type] for dic in collections_dic['stat_dic']],0) for type in collections_dic['stat_dic'][experience][set_]} for set_ in collections_dic['stat_dic'][experience]}for experience in collections_dic['stat_dic']}

    plot_PCA(latent_dic_mean, 5, "mean")
    plot_class(mut_pred_dic_mean, 5, "mean", state_dic_mean)
    plot_stat(stat_dic_mean, "mean")

def main_per_collection(collection_dic, collection_id):
    experiences = ['Unsup. Learning on Evol. MAPT, Exp.1', 'Unsup. Learning on Evol./Clin. MAPT, Exp.1', 'Unsup. Learning on Evol./Clin. MAPT, Exp.2',
                   'Unsup. Learning on Evol./Clin. MAPT, Exp.3','Unsup. Learning on Evol./Clin. MAPT, Exp.4','Semi-Sup. Learning on MAPT, Exp.1',
                   'Semi-Sup. Learning on MAPT, Exp.2', 'Semi-Sup. Learning on MAPT, Exp.3', 'Semi-Sup. Learning on MAPT, Exp.4','Unsup. Learning on Evol. MAPT, Exp.5', 'Unsup. Learning on Evol./Clin. MAPT, Exp.5', 'Unsup. Learning on Evol./Clin. MAPT, Exp.6',
                   'Unsup. Learning on Evol./Clin. MAPT, Exp.7','Unsup. Learning on Evol./Clin. MAPT, Exp.8','Semi-Sup. Learning on MAPT, Exp.5',
                   'Semi-Sup. Learning on MAPT, Exp.6', 'Semi-Sup. Learning on MAPT, Exp.7', 'Semi-Sup. Learning on MAPT, Exp.8',
                   'Unsup. Learning on Evol. BRCA1, Exp.9', 'Unsup. Learning on Evol./Clin. BRCA1, Exp.9',
                   'Unsup. Learning on Evol./Clin. BRCA1, Exp.10',
                   'Unsup. Learning on Evol./Clin. BRCA1, Exp.11',
                   'Semi-Sup. Learning on BRCA1, Exp.9',
                   'Semi-Sup. Learning on BRCA1, Exp.10', 'Semi-Sup. Learning on BRCA1, Exp.11'
                   ]
    # experiences = [
    #                'Semi-Sup. Learning on BRCA1, Exp.5',
    #                'Semi-Sup. Learning on BRCA1, Exp.6', 'Semi-Sup. Learning on BRCA1, Exp.7']
    # offset = 13
    sets = ['Training Set', 'Testing Set']
    latent_dic = {}
    mut_pred_dic = {}
    labelled_mut_dic = {}
    stat_dic = {}

    dataset_sizes = ['full','full','full','full','full','full','full','full','full','small','small','small','small','small','small','small','small','small','full','full','full','full','full','full','full']

    for exp in experiences:

        latent_dic[exp] = {}
        mut_pred_dic[exp] = {}
        labelled_mut_dic[exp] = {}
        stat_dic[exp] = {}
        for set in sets:
            latent_dic[exp][set] = {}
            mut_pred_dic[exp][set] = {}
            labelled_mut_dic[exp][set] = {}
            stat_dic[exp][set] = {}
    nb_path_training_points = [0, 26, 13, 0, 9, 26, 13, 0, 9, 0, 26, 13, 0, 9, 26, 13, 0, 9, 0, 0, 2, 2, 0, 2, 2]
    nb_npath_training_points = [ 0, 9, 4, 0, 9, 9, 4, 0, 9, 0, 9, 4, 0, 9, 9, 4, 0, 9, 0, 0, 0, 1, 0, 0, 1]
    # nb_path_training_points = [0, 2, 2]
    # nb_npath_training_points = [0, 0, 1]
    data_formats = {"MAPT" : "alz", "BRCA1" : "manual"}
    if opt.test_algo:
        dataset_files = {'full':{"MAPT" : "./datasets/P10636.a2m", "BRCA1" : "./datasets/BRCA1_small.a2m"}, 'small':{"MAPT": "./datasets/P10636.a2m", "BRCA1": "./datasets/BRCA1_small.a2m"}}

        params = {"epochs": 2, "test_every": 1, "n_pred": 2}
    else:
        dataset_files = {'full':{"MAPT" : "./datasets/P10636-8_full_b01.a2m", "BRCA1" : "./datasets/BRCA1.a2m"}, 'small':{"MAPT": "./datasets/P10636_less_data.a2m", "BRCA1": "./datasets/BRCA1.a2m"}}
        params = {"epochs": 100, "test_every": 10, "n_pred": 200}

    datahelpers = {size:{gene : DataHelper(dataset = dataset_files[size][gene], theta = 0.2, custom_dataset = True) for gene in dataset_files[size]} for size in dataset_files}

    train_functions = {"Unsup.": train_unsupervised, "Semi-Sup.": train_semsup}

    models = {"Unsup." : {gene : VAE(opt.latent_dim_M1, datahelpers['full'][gene].seqlen,
                      opt.alphabet_size, opt.enc_h1_dim, opt.enc_h2_dim, opt.dec_h1_dim, opt.dec_h2_dim, opt.dim_h1_clas,
                      opt.dim_h2_clas, opt.mu_sparse, opt.logsigma_sparse, opt.cw_inner_dimension, opt.nb_patterns,
                      opt.has_temperature, opt.has_dictionary).to(device) for gene in datahelpers['full']}, "Semi-Sup.": {gene: M2_VAE(opt.latent_dim_M1, datahelpers['full'][gene].seqlen, opt.alphabet_size, opt.enc_h1_dim, opt.enc_h2_dim, opt.dec_h1_dim,
                   opt.dec_h2_dim, 1, 2, opt.dim_h1_clas,
                   opt.dim_h2_clas, opt.mu_sparse, opt.logsigma_sparse, opt.cw_inner_dimension, opt.nb_patterns,
                   opt.has_temperature, opt.has_dictionary).to(device) for gene in datahelpers['full']}}
    optimizers = {method:{gene:torch.optim.Adam(models[method][gene].parameters()) for gene in models[method]}for method in models}
    init_states = {method:{gene:{'model': copy.deepcopy(models[method][gene].state_dict()), 'opt': copy.deepcopy(optimizers[method][gene].state_dict())} for gene in models[method]}for method in models}

    pre_process_datas = {size:{gene : preprocess_data(gene, data_formats[gene], datahelpers[size][gene]) for gene in data_formats} for size in datahelpers}

    for num_exp ,experience in enumerate(experiences):
        size = dataset_sizes[num_exp]
        print(experience)
        if "MAPT" in experience:
            gene = "MAPT"
        else:
            gene = "BRCA1"
        if "Unsup." in experience:
            method = "Unsup."
        else:
            method = "Semi-Sup."
        clinical_data = get_clinical_data_decomposition(pre_process_datas[size][gene], data_formats[gene], datahelpers[size][gene], nb_path_training_points[num_exp], nb_npath_training_points[num_exp])
        npath_test_seq = get_seq(clinical_data["npath_test"], pre_process_datas[size][gene]["healthy_seq"],
                                 focus_index=datahelpers[size][gene].focus_index, defocus_index=datahelpers[size][gene].defocus_index,
                                 data_format=data_formats[gene])
        path_test_seq = get_seq(clinical_data["path_test"], pre_process_datas[size][gene]["healthy_seq"],
                                focus_index=datahelpers[size][gene].focus_index, defocus_index=datahelpers[size][gene].defocus_index,
                                data_format=data_formats[gene])

        path_test_one_hot = get_one_hot(path_test_seq)
        npath_test_one_hot = get_one_hot(npath_test_seq)

        labelled_mut_dic[experience]['Testing Set']['Pathogenic'] = path_test_one_hot
        mut_pred_dic[experience]['Testing Set']['Pathogenic'] = {}
        latent_dic[experience]['Testing Set']['Pathogenic'] = {}
        stat_dic[experience]['Testing Set']['Pathogenic'] = {}
        labelled_mut_dic[experience]['Testing Set']['Not Pathogenic'] = npath_test_one_hot
        mut_pred_dic[experience]['Testing Set']['Not Pathogenic'] = {}
        latent_dic[experience]['Testing Set']['Not Pathogenic'] = {}
        stat_dic[experience]['Testing Set']['Not Pathogenic'] = {}

        if nb_path_training_points[num_exp] > 0:
            path_train_seq = get_seq(clinical_data["path_train"], pre_process_datas[size][gene]["healthy_seq"],
                                     focus_index=datahelpers[size][gene].focus_index, defocus_index=datahelpers[size][gene].defocus_index,
                                     data_format=data_formats[gene])
            path_train_one_hot = get_one_hot(path_train_seq)
            labelled_mut_dic[experience]['Training Set']['Pathogenic'] = path_train_one_hot
            mut_pred_dic[experience]['Training Set']['Pathogenic'] = {}
            latent_dic[experience]['Training Set']['Pathogenic'] = {}
            stat_dic[experience]['Training Set']['Pathogenic'] = {}

        if nb_npath_training_points[num_exp] > 0:
            npath_train_seq = get_seq(clinical_data["npath_train"], pre_process_datas[size][gene]["healthy_seq"],
                                      focus_index=datahelpers[size][gene].focus_index, defocus_index=datahelpers[size][gene].defocus_index,
                                      data_format=data_formats[gene])
            npath_train_one_hot = get_one_hot(npath_train_seq)
            labelled_mut_dic[experience]['Training Set']['Not Pathogenic'] = npath_train_one_hot
            mut_pred_dic[experience]['Training Set']['Not Pathogenic'] = {}
            latent_dic[experience]['Training Set']['Not Pathogenic'] = {}
            stat_dic[experience]['Training Set']['Not Pathogenic'] = {}
        train_functions[method](data_formats[gene], clinical_data, models[method][gene], optimizers[method][gene], init_states[method][gene], datahelpers[dataset_sizes[num_exp]][gene], experience, num_exp, nb_path_training_points[num_exp], nb_npath_training_points[num_exp], pre_process_datas[dataset_sizes[num_exp]][gene], latent_dic, mut_pred_dic, labelled_mut_dic, params, collection_id)

    plot_PCA(latent_dic, 5, collection_id)
    plot_class(mut_pred_dic, 5, collection_id, stat_dic)
    plot_stat(stat_dic, collection_id)
    collection_dic['labelled_mut_dic'].append(labelled_mut_dic)
    collection_dic['mut_pred_dic'].append(mut_pred_dic)
    collection_dic['latent_dic'].append(latent_dic)
    collection_dic['stat_dic'].append(stat_dic)


def train_unsupervised(data_format, clinical_data, model, optimizer, init_state, datahelper, experience, num_exp, nb_path_training_point, nb_npath_training_point, pre_process_data, latent_dic, mut_pred_dic, labelled_mut_dic, params, collection_id):
    model.load_state_dict(init_state['model'])
    optimizer.load_state_dict(init_state['opt'])

    model.train()

    if nb_npath_training_point + nb_path_training_point == 0:
        sampler = torch.utils.data.sampler.WeightedRandomSampler(datahelper.weights, datahelper.datasize)
        train_dataset = Dataset(datahelper)
        train_dataset_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, sampler=sampler)
        Neff = datahelper.Neff

    else:
        all_seq = get_seq(clinical_data["all"], pre_process_data["healthy_seq"], focus_index=datahelper.focus_index, defocus_index=datahelper.defocus_index, data_format=data_format)
        all_one_hot = get_one_hot(all_seq)
        N1 = len(datahelper.weights)
        N = len(datahelper.weights) + len(all_seq)
        weigth_human_genome = (datahelper.Neff * N / N1 - datahelper.Neff) / (N - N1)
        Neff = datahelper.Neff * N / N1
        weigth_human_genome_tsr = weigth_human_genome * torch.ones(len(all_seq))
        weights_merge = torch.cat([weigth_human_genome_tsr, datahelper.weights])

        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights_merge, N)
        train_dataset = Dataset_merge(datahelper, all_one_hot)
        train_dataset_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, sampler=sampler)

    loss_plot = {'loss': [], 'KLD_weights': [], 'KLD_latent': [], 'logpx_z': []}




    training_loop_unsupervised(params['epochs'], train_dataset_loader, optimizer, model, loss_plot, params['test_every'],
                Neff, labelled_mut_dic, mut_pred_dic,
                               pre_process_data['healthy_one_hot'], datahelper.alphabet_size, datahelper.seqlen, experience, latent_dic, params['n_pred'])

    if opt.plot_loss:
        plt.clf()
        plt.figure()
        for i, label in enumerate(loss_plot):
            y = loss_plot[label]
            plt.subplot(1, len(loss_plot), i + 1)
            plt.plot(np.arange(len(y)), y)
            plt.title(label)
        plt.suptitle("Training Losses\n {}".format(experience))
        plt.savefig(
            "./training_plots/training_loss_exp{}_{}".format(num_exp, collection_id))
        plt.close('all')

def train_semsup(data_format, clinical_data, model, optimizer, init_state, datahelper, experience, num_exp, nb_path_training_point, nb_npath_training_point, pre_process_data, latent_dic, mut_pred_dic, labelled_mut_dic, params, collection_id):
    model.load_state_dict(init_state['model'])
    optimizer.load_state_dict(init_state['opt'])

    if len(clinical_data["unlabeled"])>0:
        unlabeled_seq = get_seq(clinical_data["unlabeled"], pre_process_data["healthy_seq"], focus_index=datahelper.focus_index, defocus_index=datahelper.defocus_index, data_format=data_format)
        unlabeled_hot = get_one_hot(unlabeled_seq)

        N1 = len(datahelper.weights)
        N = len(datahelper.weights) + len(unlabeled_seq)
        weigth_human_genome = (datahelper.Neff * N / N1 - datahelper.Neff) / (N - N1)
        Neff = datahelper.Neff * N / N1
        weigth_human_genome_tsr = weigth_human_genome * torch.ones(len(unlabeled_seq))
        weights_merge = torch.cat([weigth_human_genome_tsr, datahelper.weights])

        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights_merge, N)
        train_dataset = Dataset_merge(datahelper, unlabeled_hot)
        unlabelled_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, sampler=sampler)

    else:
        sampler = torch.utils.data.sampler.WeightedRandomSampler(datahelper.weights, datahelper.datasize)
        train_dataset = Dataset(datahelper)
        unlabelled_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, sampler=sampler)
        Neff = datahelper.Neff

    labeled_seq = get_seq(clinical_data["labeled"], pre_process_data["healthy_seq"], focus_index=datahelper.focus_index, defocus_index=datahelper.defocus_index, data_format=data_format)

    no_label = len(labeled_seq) == 0

    if not no_label:
        labeled_hot = get_one_hot(labeled_seq)

    test_seq = get_seq(clinical_data["test"], pre_process_data["healthy_seq"], focus_index=datahelper.focus_index, defocus_index=datahelper.defocus_index, data_format=data_format)
    test_hot = get_one_hot(test_seq)
    test_dataset = Dataset_labelled(test_hot, clinical_data["test_labels"])
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True)

    if not no_label:
        labeled_dataset = Dataset_labelled(labeled_hot, clinical_data["labels"])
        labeled_loader = DataLoader(labeled_dataset, batch_size=opt.batch_size, shuffle=True)

    # first option: label+jackhammer problem: some label would lose pathogenic feature
    # second option: jackhammer (remove labeled mut without mutation anymore) try this one
    # third option: label (keep position that jack want to remove)

    if not no_label:
        loss_plot = {'u_loss': [], 'clas_loss': [], 'l_loss': [], 'kld': []}

        clas_plot = {'test_clas_loss': [], 'test_accuracy (%)': [], 'train_clas_loss': [], 'train_accuracy (%)': []}

        training_loop_with_label(params['epochs'], labeled_loader, unlabelled_loader, optimizer, model, loss_plot, params['test_every'],
                         test_loader, clas_plot, Neff, labelled_mut_dic, mut_pred_dic,
                         pre_process_data['healthy_one_hot'], datahelper.alphabet_size, datahelper.seqlen, experience, latent_dic, params['n_pred'])

    else:
        loss_plot = {'u_loss': [], 'kld': []}

        clas_plot = {'test_clas_loss': [], 'test_accuracy (%)': []}
        training_loop_without_label(params['epochs'], unlabelled_loader, optimizer, model, loss_plot, params['test_every'],
                            test_loader, clas_plot, Neff, labelled_mut_dic, mut_pred_dic,
                                    pre_process_data['healthy_one_hot'], datahelper.alphabet_size, datahelper.seqlen, experience, latent_dic, params['n_pred'])

    if opt.plot_loss:
        plt.clf()
        plt.figure()
        for i, label in enumerate(loss_plot):
            y = loss_plot[label]
            plt.subplot(1, len(loss_plot), i + 1)
            plt.plot(np.arange(len(y)), y)
            plt.title(label)
        plt.suptitle("Training Losses\n{}".format(experience))
        plt.savefig(
            "./training_plots/training_losses_exp{}_{}".format(num_exp, collection_id))
        plt.close('all')

    plt.clf()
    plt.figure()
    for i, label in enumerate(clas_plot):
        y = clas_plot[label]
        plt.subplot(1, len(clas_plot), i + 1)
        plt.plot(np.arange(len(y)), y)
        plt.title(label)
    plt.suptitle(
        "Classification Loss\n{}".format(experience))
    plt.savefig(
        "./classification_plots/classification_loss_exp{}_{}".format(num_exp,collection_id))
    plt.close('all')


def training_loop_with_label(epochs_, labeled_loader, unlabelled_loader, optimizer, model, loss_plot, test_every, labeled_test_loader, clas_plot, new_Neff, labelled_mut, path_plot, healthy_one_hot, alphabet_size, seqlen, exp, latent, n_pred_iterations):
    for epoch in range(0, epochs_ + 1):
        u_losse = []
        clas_losse = []
        l_losse = []
        klde = []
        for (l_batch, l_labels), u_batch in create_joined_dataset(labeled_loader, unlabelled_loader):
            # print(k)
            # k=1
            u_batch = u_batch.to(device)
            l_batch = l_batch.to(device)
            l_labels = l_labels.to(device)
            optimizer.zero_grad()
            # labeled mutations
            mu, logsigma, px_zy, logpx_zy, qy_x, logqy_x, y, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l = model(
                l_batch, l_labels)
            loss_labeled = -labeled_mut_loss(logpx_zy, mu, logsigma, y, logqy_x)
            loss_plot['l_loss'].append(loss_labeled.item())
            loss_plot['clas_loss'].append(classification_loss(l_labels.float(), logqy_x).item())
            kld_l = kld_latent_theano(mu, logsigma).mean().item()
            l_losse.append(logpx_zy.mean().item())
            clas_losse.append(classification_loss(l_labels.float(), logqy_x).item())

            # unlabeled mutations
            mu, logsigma, px_zy, logpx_zy, qy_x, logqy_x, y, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l = model(
                u_batch)
            loss_unlabelled = -unlabelled_mut_loss(logpx_zy, mu, logsigma, qy_x, logqy_x, y) + sparse_theano(mu_W1, logsigma_W1,
                                                                                                 mu_b1, logsigma_b1,
                                                                                                 mu_W2, logsigma_W2,
                                                                                                 mu_b2, logsigma_b2,
                                                                                                 mu_W3, logsigma_W3,
                                                                                                 mu_b3, logsigma_b3,
                                                                                                 mu_S, logsigma_S, mu_C,
                                                                                                 logsigma_C, mu_l,
                                                                                                 logsigma_l) / new_Neff
            loss_plot['u_loss'].append(loss_unlabelled.item())
            kld_u = kld_latent_theano(mu, logsigma).mean().item()
            loss_plot['kld'].append(kld_l + kld_u)
            klde.append(kld_l + kld_u)
            u_losse.append(logpx_zy.mean().item())

            loss = loss_labeled + loss_unlabelled
            loss.backward()
            optimizer.step()
        if epoch % test_every == 0:
            model.eval()
            test_loss = 0
            correct = 0
            train_loss = 0
            train_correct = 0
            with torch.no_grad():
                for (l_batch, l_labels) in labeled_test_loader:
                    l_batch, l_labels = l_batch.to(device), l_labels.to(device)
                    _, _, _, _, qy_x, logqy_x = model(l_batch)[:6]
                    maxx = qy_x.max(dim=-1, keepdim=True)[0]
                    pred = torch.eq(qy_x, maxx).float() / 2
                    test_loss += (l_labels.float() * logqy_x.float()).sum().item()
                    correct += pred.eq(l_labels.float().view_as(pred)).sum().item()
                clas_plot['test_clas_loss'].append(test_loss / len(labeled_test_loader.dataset))
                clas_plot['test_accuracy (%)'].append(100 * correct / len(labeled_test_loader.dataset))
                for (l_batch, l_labels) in labeled_loader:
                    l_batch, l_labels = l_batch.to(device), l_labels.to(device)
                    _, _, _, _, qy_x, logqy_x = model(l_batch)[:6]
                    maxx = qy_x.max(dim=-1, keepdim=True)[0]
                    pred = torch.eq(qy_x, maxx).float() / 2
                    train_loss += (l_labels.float() * logqy_x.float()).sum().item()
                    train_correct += pred.eq(l_labels.float().view_as(pred)).sum().item()
                clas_plot['train_clas_loss'].append(train_loss / len(labeled_loader.dataset))
                clas_plot['train_accuracy (%)'].append(100 * train_correct / len(labeled_loader.dataset))

                for set in labelled_mut[exp]:
                    for path in labelled_mut[exp][set]:
                        latent[exp][set][path]['Epoch {}'.format(epoch)] = get_latent(labelled_mut[exp][set][path], model)
                        path_plot[exp][set][path]['Epoch {}'.format(epoch)] = pred_from_onehot(
                            labelled_mut[exp][set][path], healthy_one_hot, model, alphabet_size, seqlen,
                            semi_supervised=True, N_pred_iterations = n_pred_iterations)
            model.train()

def training_loop_unsupervised(epochs_, unlabelled_loader, optimizer, model, loss_plot, test_every, new_Neff, labelled_mut_dic, path_plot, healthy_one_hot, alphabet_size, seqlen, exp, latent, n_pred_iterations):

    for epoch in range(0, epochs_ + 1):
        for u_batch in unlabelled_loader:
            u_batch = u_batch.to(device)

            optimizer.zero_grad()

            # unlabeled mutations
            mu, logsigma, px_z, logpx_z, z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l = model(
                u_batch)
            loss = -loss_theano(mu, logsigma, z, px_z, logpx_z, new_Neff, 1.0, mu_W1, logsigma_W1, mu_b1,
                                logsigma_b1, mu_W2,
                                logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S,
                                logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l)
            loss_plot['loss'].append(loss.item())
            loss_plot['KLD_weights'].append(
                (1.0 * sparse_theano(mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2,
                                     logsigma_b2,
                                     mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C,
                                     logsigma_C,
                                     mu_l, logsigma_l) / new_Neff).item())
            loss_plot['KLD_latent'].append((1.0 * kld_latent_theano(mu, logsigma).mean()).item())
            loss_plot['logpx_z'].append(logpx_z.mean().item())

            loss.backward()
            optimizer.step()
        if epoch % test_every == 0:
            model.eval()
            with torch.no_grad():
                for set in labelled_mut_dic[exp]:
                    for path in labelled_mut_dic[exp][set]:
                        latent[exp][set][path]['Epoch {}'.format(epoch)] = get_latent(labelled_mut_dic[exp][set][path],model)
                        path_plot[exp][set][path]['Epoch {}'.format(epoch)] = pred_from_onehot(labelled_mut_dic[exp][set][path], healthy_one_hot, model, alphabet_size, seqlen, N_pred_iterations = n_pred_iterations)
            model.train()



def training_loop_without_label(epochs_, unlabelled_loader, optimizer, model, loss_plot, test_every, labeled_test_loader, clas_plot, new_Neff, labelled_mut, path_plot, healthy_one_hot, alphabet_size, seqlen, exp, latent, n_pred_iterations):
    for epoch in range(0, epochs_ + 1):
        u_losse = []
        klde = []
        for u_batch in unlabelled_loader:
            u_batch = u_batch.to(device)

            optimizer.zero_grad()
            mu, logsigma, px_zy, logpx_zy, qy_x, logqy_x, y, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l = model(
                u_batch)
            loss_unlabelled = -unlabelled_mut_loss(logpx_zy, mu, logsigma, qy_x, logqy_x, y) + sparse_theano(mu_W1, logsigma_W1,
                                                                                                 mu_b1, logsigma_b1,
                                                                                                 mu_W2, logsigma_W2,
                                                                                                 mu_b2, logsigma_b2,
                                                                                                 mu_W3, logsigma_W3,
                                                                                                 mu_b3, logsigma_b3,
                                                                                                 mu_S, logsigma_S, mu_C,
                                                                                                 logsigma_C, mu_l,
                                                                                                 logsigma_l) / new_Neff
            loss_plot['u_loss'].append(loss_unlabelled.item())
            kld_u = kld_latent_theano(mu, logsigma).mean().item()
            loss_plot['kld'].append(kld_u)
            klde.append(kld_u)
            u_losse.append(logpx_zy.mean().item())

            loss = loss_unlabelled
            loss.backward()
            optimizer.step()
        if epoch % test_every == 0:
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for (l_batch, l_labels) in labeled_test_loader:
                    l_batch, l_labels = l_batch.to(device), l_labels.to(device)
                    _, _, _, _, qy_x, logqy_x = model(l_batch)[:6]
                    maxx = qy_x.max(dim=-1, keepdim=True)[0]
                    pred = torch.eq(qy_x, maxx).float() / 2
                    test_loss += (l_labels.float() * logqy_x.float()).sum().item()
                    correct += pred.eq(l_labels.float().view_as(pred)).sum().item()
                clas_plot['test_clas_loss'].append(test_loss / len(labeled_test_loader.dataset))
                clas_plot['test_accuracy (%)'].append(100 * correct / len(labeled_test_loader.dataset))


                for set in labelled_mut[exp]:
                    # print(set)
                    for path in labelled_mut[exp][set]:
                        latent[exp][set][path]['Epoch {}'.format(epoch)] = get_latent(labelled_mut[exp][set][path],
                                                                                      model)
                        path_plot[exp][set][path]['Epoch {}'.format(epoch)] = pred_from_onehot(
                            labelled_mut[exp][set][path], healthy_one_hot, model, alphabet_size, seqlen,
                            semi_supervised=True, N_pred_iterations=n_pred_iterations)
            model.train()

def plot_class(path_plot, plots_per_line, collection_id, stat_dic):
    for j,exp in enumerate(path_plot):
        for h,set in enumerate(path_plot[exp]):
            # print(set)
            for k,path in enumerate(path_plot[exp][set]):

                # print(path)
                plt.clf()
                plt.figure() #default size value: 8.0, 6.0
                for i , epoch in enumerate(path_plot[exp][set][path]):

                    y = path_plot[exp][set][path][epoch]
                    mu = np.mean(y)
                    ls = np.std(y)
                    x = np.linspace(mu - 3 * ls, mu + 3 * ls, max(1000, 6*ls * 100))
                    # print(len(path_plot[method][set][path]))
                    # print(int(len(path_plot[method][set][path])/plots_per_line))
                    plt.subplot(math.ceil(len(path_plot[exp][set][path])/plots_per_line), plots_per_line, i + 1)
                    plt.plot(x, stats.norm.pdf(x, mu, ls))
                    z = np.zeros_like(y)
                    plt.scatter(y, z)
                    plt.title(epoch)
                plt.suptitle("Labelled Mutation Prediction\n{}\n{}\n{}".format(exp, set, path))
                plt.savefig(
                    "./mutation_plots/lmp_exp{}_set{}_path{}_{}".format(j,h,k, collection_id))
                plt.close('all')
                plt.clf()
                plt.figure(figsize=(8.0*plots_per_line,6.0*math.ceil(len(path_plot[exp][set][path])/plots_per_line)))  # default size value: 8.0, 6.0
                for i, epoch in enumerate(path_plot[exp][set][path]):
                    y = path_plot[exp][set][path][epoch]
                    mu = np.mean(y)
                    ls = np.std(y)
                    x = np.linspace(mu - 3 * ls, mu + 3 * ls, max(1000, 6 * ls * 100))
                    plt.subplot(math.ceil(len(path_plot[exp][set][path]) / plots_per_line), plots_per_line, i + 1)
                    plt.plot(x, stats.norm.pdf(x, mu, ls))
                    z = np.zeros_like(y)
                    plt.scatter(y, z)
                    plt.title(epoch)
                plt.suptitle("Labelled Mutation Prediction\n{}\n{}\n{}".format(exp, set, path))
                plt.savefig(
                    "./mutation_plots/lmp_exp{}_set{}_path{}_hd_{}".format(j,h,k, collection_id))
                plt.close('all')

    for j,exp in enumerate(path_plot):
        print(exp)
        for h,set in enumerate(path_plot[exp]):
            print(set)
            if 'Pathogenic' in path_plot[exp][set] and 'Not Pathogenic' in path_plot[exp][set]:
                plt.clf()
                plt.figure()
                for i , epoch in enumerate(path_plot[exp][set]['Pathogenic']):
                    y_path = path_plot[exp][set]['Pathogenic'][epoch]
                    y_npath = path_plot[exp][set]['Not Pathogenic'][epoch]
                    mu_path = np.mean(y_path)
                    ls_path = np.std(y_path)
                    mu_npath = np.mean(y_npath)
                    ls_npath = np.std(y_npath)
                    x = np.linspace(min(mu_npath - 3 * ls_npath, mu_path - 3 * ls_path),
                        max(mu_npath + 3 * ls_npath, mu_path + 3 * ls_path), max(1000, (
                        max(mu_npath + 3 * ls_npath, mu_path + 3 * ls_path) - min(mu_npath - 3 * ls_npath,
                                                                                  mu_path - 3 * ls_path)) * 100)
                        )
                    plt.subplot(math.ceil(len(path_plot[exp][set]['Pathogenic']) / plots_per_line), plots_per_line, i + 1)
                    plt.plot(x, stats.norm.pdf(x, mu_npath, ls_npath), label="Not Pathogenic")
                    if len(y_npath)==1:
                        z = np.zeros_like(y_npath)
                        plt.scatter(y_npath, z)
                    plt.plot(x, stats.norm.pdf(x, mu_path, ls_path), label="Pathogenic")
                    if len(y_path)==1:
                        z2 = np.zeros_like(y_path)
                        plt.scatter(y_path, z2)
                    plt.legend(title="Mutation Type:")
                    plt.title(epoch)
                plt.suptitle("Labelled Mutation Prediction\n{}\n{}".format(exp, set))
                plt.savefig(
                    "./mutation_plots/lbm_exp{}_set{}_{}".format(j,h,collection_id))
                plt.close('all')
                plt.clf()
                stat_dic[exp][set]['mu_path'] = mu_path
                stat_dic[exp][set]['ls_path'] = ls_path
                stat_dic[exp][set]['mu_npath'] = mu_npath
                stat_dic[exp][set]['ls_npath'] = ls_npath
                print(mu_path)
                print(ls_path)
                print(mu_npath)
                print(ls_npath)
                try:
                    print(JSD(mu_path,ls_path, mu_npath,ls_npath)) #should be log(ls...)
                except:
                    print("overflow")
                plt.figure(figsize=(8.0*plots_per_line,6.0*math.ceil(len(path_plot[exp][set]['Pathogenic'])/plots_per_line)))
                for i, epoch in enumerate(path_plot[exp][set]['Pathogenic']):
                    y_path = path_plot[exp][set]['Pathogenic'][epoch]
                    y_npath = path_plot[exp][set]['Not Pathogenic'][epoch]
                    mu_path = np.mean(y_path)
                    ls_path = np.std(y_path)
                    mu_npath = np.mean(y_npath)
                    ls_npath = np.std(y_npath)
                    x = np.linspace(min(mu_npath - 3 * ls_npath, mu_path - 3 * ls_path),
                                    max(mu_npath + 3 * ls_npath, mu_path + 3 * ls_path), max(1000, (
                                max(mu_npath + 3 * ls_npath, mu_path + 3 * ls_path) - min(mu_npath - 3 * ls_npath,
                                                                                          mu_path - 3 * ls_path)) * 100)
                                    )
                    plt.subplot(math.ceil(len(path_plot[exp][set]['Pathogenic']) / plots_per_line), plots_per_line, i + 1)
                    plt.plot(x, stats.norm.pdf(x, mu_npath, ls_npath), label="Not Pathogenic")
                    plt.plot(x, stats.norm.pdf(x, mu_path, ls_path), label="Pathogenic")
                    plt.legend(title="Mutation Type:")
                    plt.title(epoch)
                plt.suptitle("Labelled Mutation Prediction\n{}\n{}".format(exp, set))
                plt.savefig(
                    "./mutation_plots/lmp_exp{}_set{}_hd_{}".format(j,h, collection_id))
                plt.close('all')
                
def plot_stat(stat_dic, collection_id):
    unsup = []
    unsup_index = []
    unsupc = []
    unsupc_index = []
    ssup = []
    ssup_index = []

    for index, experience in enumerate(stat_dic):
        if "BRCA1" in experience and len(stat_dic[experience]['Testing Set']) > 1:
            if "Evol./Clin." in experience:
                unsupc.append(
                    abs(stat_dic[experience]['Testing Set']['mu_path'] - stat_dic[experience]['Testing Set']['mu_npath']))
                unsupc_index.append(index)
            elif "Unsup." in experience:
                unsup.append(
                    abs(stat_dic[experience]['Testing Set']['mu_path'] - stat_dic[experience]['Testing Set']['mu_npath']))
                unsup_index.append(index)
            else:
                ssup.append(
                    abs(stat_dic[experience]['Testing Set']['mu_path'] - stat_dic[experience]['Testing Set']['mu_npath']))
                ssup_index.append(index)
    plt.clf()
    plt.scatter(unsupc_index, unsupc,
                label="Unsup. Evol./Clin.")
    plt.scatter(unsup_index, unsup, label="Unsup. Evol.")
    plt.scatter(ssup_index, ssup, label="Semi-Sup.")
    plt.legend(title="Learning Type:")
    plt.ylabel("mu_path-mu_npath")
    plt.xlabel("Experience")
    plt.title("Difference between means, BRCA1, Testing Set")
    plt.savefig("means_dif_BRCA1_test_{}".format(collection_id))
    plt.close('all')

    unsup = []
    unsup_index = []
    unsupc = []
    unsupc_index = []
    ssup = []
    ssup_index = []

    for index, experience in enumerate(stat_dic):
        if "MAPT" in experience and len(stat_dic[experience]['Testing Set']) > 1:
            if "Evol./Clin." in experience:
                unsupc.append(
                    abs(stat_dic[experience]['Testing Set']['mu_path'] - stat_dic[experience]['Testing Set']['mu_npath']))
                unsupc_index.append(index)
            elif "Unsup." in experience:
                unsup.append(
                    abs(stat_dic[experience]['Testing Set']['mu_path'] - stat_dic[experience]['Testing Set']['mu_npath']))
                unsup_index.append(index)
            else:
                ssup.append(
                    abs(stat_dic[experience]['Testing Set']['mu_path'] - stat_dic[experience]['Testing Set']['mu_npath']))
                ssup_index.append(index)
    plt.clf()
    plt.scatter(unsupc_index, unsupc,
                label="Unsup. Evol./Clin.")
    plt.scatter(unsup_index, unsup, label="Unsup. Evol.")
    plt.scatter(ssup_index, ssup, label="Semi-Sup.")
    plt.legend(title="Learning Type:")
    plt.ylabel("mu_path-mu_npath")
    plt.xlabel("Experience")
    plt.title("Difference between means, MAPT, Testing Set")
    plt.savefig("means_dif_MAPT_test_{}".format(collection_id))
    plt.close('all')

    unsup = []
    unsup_index = []
    unsupc = []
    unsupc_index = []
    ssup = []
    ssup_index = []

    for index, experience in enumerate(stat_dic):
        if "MAPT" in experience and len(stat_dic[experience]['Training Set']) > 1:
            if "Evol./Clin." in experience:
                unsupc.append(abs(
                    stat_dic[experience]['Training Set']['mu_path'] - stat_dic[experience]['Training Set']['mu_npath']))
                unsupc_index.append(index)
            elif "Unsup." in experience:
                unsup.append(
                    abs(stat_dic[experience]['Training Set']['mu_path'] - stat_dic[experience]['Training Set'][
                        'mu_npath']))
                unsup_index.append(index)
            else:
                ssup.append(
                    abs(stat_dic[experience]['Training Set']['mu_path'] - stat_dic[experience]['Training Set'][
                        'mu_npath']))
                ssup_index.append(index)
    plt.clf()
    plt.scatter(unsupc_index, unsupc,
                label="Unsup. Evol./Clin.")
    plt.scatter(unsup_index, unsup, label="Unsup. Evol.")
    plt.scatter(ssup_index, ssup, label="Semi-Sup.")
    plt.legend(title="Learning Type:")
    plt.ylabel("mu_path-mu_npath")
    plt.xlabel("Experience")
    plt.title("Difference between means, MAPT, Training Set")
    plt.savefig("means_dif_MAPT_train_{}".format(collection_id))
    plt.close('all')

    unsup = []
    unsup_index = []
    unsupc = []
    unsupc_index = []
    ssup = []
    ssup_index = []

    for index, experience in enumerate(stat_dic):
        if "BRCA1" in experience and len(stat_dic[experience]['Training Set']) > 1:
            print(stat_dic[experience]['Training Set'])
            if "Evol./Clin." in experience:
                unsupc.append(abs(
                    stat_dic[experience]['Training Set']['mu_path'] - stat_dic[experience]['Training Set']['mu_npath']))
                unsupc_index.append(index)
            elif "Unsup." in experience:
                unsup.append(
                    abs(stat_dic[experience]['Training Set']['mu_path'] - stat_dic[experience]['Training Set'][
                        'mu_npath']))
                unsup_index.append(index)
            else:
                ssup.append(
                    abs(stat_dic[experience]['Training Set']['mu_path'] - stat_dic[experience]['Training Set'][
                        'mu_npath']))
                ssup_index.append(index)
    plt.clf()
    plt.scatter(unsupc_index, unsupc,
                label="Unsup. Evol./Clin.")
    plt.scatter(unsup_index, unsup, label="Unsup. Evol.")
    plt.scatter(ssup_index, ssup, label="Semi-Sup.")
    plt.legend(title="Learning Type:")
    plt.ylabel("mu_path-mu_npath")
    plt.xlabel("Experience")
    plt.title("Difference between means, BRCA1, Training Set")
    plt.savefig("means_dif_BRCA1_train_{}".format(collection_id))
    plt.close('all')

    def distance_with_std(mu1, std1, mu2, std2):
        d1 = abs((mu1 - std1) - (mu2 - std2))
        d2 = abs((mu1 + std1) - (mu2 - std2))
        d3 = abs((mu1 - std1) - (mu2 + std2))
        d4 = abs((mu1 + std1) - (mu2 + std2))
        return min(d1, d2, d3, d4)

    def distance_with_std_exp(experience):
        return distance_with_std(stat_dic[experience]['Testing Set']['mu_path'],
                                 stat_dic[experience]['Testing Set']['ls_path'],
                                 stat_dic[experience]['Testing Set']['mu_npath'],
                                 stat_dic[experience]['Testing Set']['ls_npath'])

    unsup = []
    unsup_index = []
    unsupc = []
    unsupc_index = []
    ssup = []
    ssup_index = []

    for index, experience in enumerate(stat_dic):
        if "BRCA1" in experience and len(stat_dic[experience]['Testing Set']) > 1:
            if "Evol./Clin." in experience:
                unsupc.append(distance_with_std_exp(experience))
                unsupc_index.append(index)
            elif "Unsup." in experience:
                unsup.append(distance_with_std_exp(experience))
                unsup_index.append(index)
            else:
                ssup.append(distance_with_std_exp(experience))
                ssup_index.append(index)
    plt.clf()
    plt.scatter(unsupc_index, unsupc,
                label="Unsup. Evol./Clin.")
    plt.scatter(unsup_index, unsup, label="Unsup. Evol.")
    plt.scatter(ssup_index, ssup, label="Semi-Sup.")
    plt.legend(title="Learning Type:")
    plt.ylabel("mu_path-mu_npath")
    plt.xlabel("Experience")
    plt.title("Difference between means with std, BRCA1, Testing Set")
    plt.savefig("means_std_dif_BRCA1_test_{}".format(collection_id))
    plt.close('all')

    unsup = []
    unsup_index = []
    unsupc = []
    unsupc_index = []
    ssup = []
    ssup_index = []

    for index, experience in enumerate(stat_dic):
        if "MAPT" in experience and len(stat_dic[experience]['Testing Set']) > 1:
            if "Evol./Clin." in experience:
                unsupc.append(distance_with_std_exp(experience))
                unsupc_index.append(index)
            elif "Unsup." in experience:
                unsup.append(distance_with_std_exp(experience))
                unsup_index.append(index)
            else:
                ssup.append(distance_with_std_exp(experience))
                ssup_index.append(index)
    plt.clf()
    plt.scatter(unsupc_index, unsupc,
                label="Unsup. Evol./Clin.")
    plt.scatter(unsup_index, unsup, label="Unsup. Evol.")
    plt.scatter(ssup_index, ssup, label="Semi-Sup.")
    plt.legend(title="Learning Type:")
    plt.ylabel("mu_path-mu_npath")
    plt.xlabel("Experience")
    plt.title("Difference between means with std, MAPT, Testing Set")
    plt.savefig("means_std_dif_MAPT_test_{}".format(collection_id))
    plt.close('all')

    unsup = []
    unsup_index = []
    unsupc = []
    unsupc_index = []
    ssup = []
    ssup_index = []

    for index, experience in enumerate(stat_dic):
        if "MAPT" in experience and len(stat_dic[experience]['Training Set']) > 1:
            if "Evol./Clin." in experience:
                unsupc.append(distance_with_std_exp(experience))
                unsupc_index.append(index)
            elif "Unsup." in experience:
                unsup.append(distance_with_std_exp(experience))
                unsup_index.append(index)
            else:
                ssup.append(distance_with_std_exp(experience))
                ssup_index.append(index)
    plt.clf()
    plt.scatter(unsupc_index, unsupc,
                label="Unsup. Evol./Clin.")
    plt.scatter(unsup_index, unsup, label="Unsup. Evol.")
    plt.scatter(ssup_index, ssup, label="Semi-Sup.")
    plt.legend(title="Learning Type:")
    plt.ylabel("mu_path-mu_npath")
    plt.xlabel("Experience")
    plt.title("Difference between means with std, MAPT, Training Set")
    plt.savefig("means_std_dif_MAPT_train_{}".format(collection_id))
    plt.close('all')

    unsup = []
    unsup_index = []
    unsupc = []
    unsupc_index = []
    ssup = []
    ssup_index = []

    for index, experience in enumerate(stat_dic):
        if "BRCA1" in experience and len(stat_dic[experience]['Training Set']) > 1:
            if "Evol./Clin." in experience:
                unsupc.append(distance_with_std_exp(experience))
                unsupc_index.append(index)
            elif "Unsup." in experience:
                unsup.append(distance_with_std_exp(experience))
                unsup_index.append(index)
            else:
                ssup.append(distance_with_std_exp(experience))
                ssup_index.append(index)
    plt.clf()
    plt.scatter(unsupc_index, unsupc,
                label="Unsup. Evol./Clin.")
    plt.scatter(unsup_index, unsup, label="Unsup. Evol.")
    plt.scatter(ssup_index, ssup, label="Semi-Sup.")
    plt.legend(title="Learning Type:")
    plt.ylabel("mu_path-mu_npath")
    plt.xlabel("Experience")
    plt.title("Difference between means with std, BRCA1, Training Set")
    plt.savefig("means_std_dif_BRCA1_train_{}".format(collection_id))
    plt.close('all')

def plot_PCA(latent_dic, plots_per_line, collection_id):
    pca = PCA(n_components=2)

    latent_dic_PCA = {}
    for exp in latent_dic:
        latent_dic_PCA[exp] = {}
        for set in latent_dic[exp]:
            latent_dic_PCA[exp][set] = {}
            if len(latent_dic[exp][set])==0:
                continue
            elif len(latent_dic[exp][set])==2:
                epochs = latent_dic[exp][set]['Not Pathogenic'].keys()
                latent_dic_PCA[exp][set]['Not Pathogenic'] = {}
                latent_dic_PCA[exp][set]['Pathogenic'] = {}
                for epoch in latent_dic[exp][set]['Not Pathogenic']:
                    all_mut = np.concatenate((latent_dic[exp][set]['Not Pathogenic'][epoch], latent_dic[exp][set]['Pathogenic'][epoch]))
                    all_mut_PCA = pca.fit_transform(all_mut)
                    latent_dic_PCA[exp][set]['Not Pathogenic'][epoch] = all_mut_PCA[:len(latent_dic[exp][set]['Not Pathogenic'][epoch])]
                    latent_dic_PCA[exp][set]['Pathogenic'][epoch] = all_mut_PCA[len(
                        latent_dic[exp][set]['Not Pathogenic'][epoch]):]
            else:
                for path in latent_dic[exp][set]:
                    latent_dic_PCA[exp][set][path]={}
                    for epoch in latent_dic[exp][set][path]:
                        latent_dic_PCA[exp][set][path][epoch] = pca.fit_transform(latent_dic[exp][set][path][epoch])
    for j,exp  in enumerate(latent_dic_PCA):
        for h,set in enumerate(latent_dic_PCA[exp]):
            if len(latent_dic_PCA[exp][set])==0:
                continue
            plt.clf()
            plt.figure()  # default size value: 8.0, 6.0
            for i, epoch in enumerate(epochs):
                plt.subplot(math.ceil(len(epochs) / plots_per_line), plots_per_line, i + 1)
                for path in latent_dic_PCA[exp][set]:
                    plt.scatter(latent_dic_PCA[exp][set][path][epoch][:,0], latent_dic_PCA[exp][set][path][epoch][:,1], label = path)
                plt.legend(title="Mutation Type:")
                plt.title(epoch)
            plt.suptitle("Latent Space (PCA) \n {}\n{}".format(exp, set))
            plt.savefig(
                "./latent_plots/latent_exp{}_set{}_{}".format(j, h, collection_id))
            plt.close('all')
    for j,exp  in enumerate(latent_dic_PCA):
        for h,set in enumerate(latent_dic_PCA[exp]):
            if len(latent_dic_PCA[exp][set])==0:
                continue
            plt.clf()
            plt.figure(figsize=(8.0*plots_per_line,6.0*math.ceil(len(epochs)/plots_per_line)))
            for i, epoch in enumerate(epochs):
                plt.subplot(math.ceil(len(epochs) / plots_per_line), plots_per_line, i + 1)
                for path in latent_dic_PCA[exp][set]:
                    plt.scatter(latent_dic_PCA[exp][set][path][epoch][:,0], latent_dic_PCA[exp][set][path][epoch][:,1], label = path)
                plt.legend(title="Mutation Type:")
                plt.title(epoch)
            plt.suptitle("Latent Space (PCA) \n {}\n{}".format(exp, set))
            plt.savefig(
                "./latent_plots/latent_exp{}_set{}_hd_{}".format(j, h, collection_id))
            plt.close('all')

def get_latent(mut_oh, model):
    mut_oh = torch.Tensor(mut_oh).to(device)
    mu = model.latent(mut_oh).cpu().numpy()
    return mu


if __name__ == "__main__":
    main()
