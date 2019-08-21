###
'''
April 2019
Code by: Arnaud Fickinger
'''
###

import torch
# torch.manual_seed(42)
import os
from collections import defaultdict
from loss import *
from bs4 import BeautifulSoup
import requests
import numpy as np
from sklearn.decomposition import PCA

# do a pca for latent space on labelled test dataset:
# pca = PCA(n_components=2)
# principalComponents = pca.fit_transform(x)
# Parameters:
# X : array-like, shape (n_samples, n_features)
# Returns:
# X_new : array-like, shape (n_samples, n_components)

from options import Options

opt = Options().parse()

def sample_diag_gaussian(mu, logvar): #reparametrization trick
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def sample_diag_gaussian_original(mu, logsigma): #reparametrization trick
    std = torch.exp(logsigma)
    eps = torch.randn_like(std)
    return mu + eps * std

def _anneal(update_num):
    """ Anneal the KL if using annealing"""
    # If true, return first, else return second
    KL_scale = torch.where(update_num < opt.warm_up, update_num/opt.warm_up, 1.0)
    return KL_scale

def log_gaussian_logvar(x, mu, logvar):
    return float(-0.5 * np.log(2 * np.pi)) - 0.5*logvar - (x - mu).pow(2) / logvar.exp()

def log_gaussian_logsigma(x, mu, logsigma):
    return float(-0.5 * np.log(2 * np.pi)) - logsigma - (x - mu).pow(2) / torch.exp(logsigma)

def embeed_feature(feature):
    if feature == 'NotPathogenic':
        return 'NotPathogenic'
    elif feature == 'UnclearPathogenicity':
        return None
    else:
        return 'Pathogenic'

def preprocess_data(gene, data_format, datahelper):
    if data_format == "alz":
        mutations, features, pathogenicity, max_rep_disease, healthy_seq = preprocess_get_label(gene)
        return {"mutations": mutations, "features": features, "pathogenicity": pathogenicity, "max_rep_disease": max_rep_disease, "healthy_seq": healthy_seq, "healthy_one_hot": get_one_hot([[healthy_seq[ix] for ix in datahelper.focus_index]])}
    else:
        healthy_seq = "MDLSALRVEEVQNVINAMQKILECPICLELIKEPVSTKCDHIFCKFCMLKLLNQKKGPSQCPLCKNDITKRSLQESTRFSQLVEELLKIICAFQLDTGLEYANSYNFAKKENNSPEHLKDEVSIIQSMGYRNRAKRLLQSEPENPSLQETSLSVQLSNLGTVRTLRTKQRIQPQKTSVYIELGSDSSEDTVNKATYCSVG"
        return {"healthy_seq": healthy_seq, "healthy_one_hot": get_one_hot([[healthy_seq[ix] for ix in datahelper.focus_index]]),
                "npath_mutations": ['K45Q', 'D67Y'], "path_mutations": ['L22S', 'T37K', 'C39R', 'H41R', 'C44S', 'C44Y'], "very_path_mutations":['C61G'] }


def get_clinical_data_decomposition(pre_process_data, data_format, datahelper, nb_path_training_point, nb_npath_training_point):
    if data_format == "alz":
        all, unlabeled, labeled, labels, test, test_labels, npath_test, path_test, npath_train, path_train = get_label(pre_process_data["mutations"], pre_process_data["features"], pre_process_data["pathogenicity"], pre_process_data["max_rep_disease"], datahelper.defocus_index, nb_path_training_point, nb_npath_training_point)
        return {"all": all, "unlabeled": unlabeled, "labeled": labeled, "labels": labels, "test" : test, "test_labels": test_labels, "npath_test": npath_test, "path_test" : path_test, "npath_train" : npath_train, "path_train" : path_train}
    else:
        npath_train = pre_process_data["npath_mutations"][:nb_npath_training_point]
        path_train = pre_process_data["path_mutations"][:nb_path_training_point]
        all_lab_train_seq = npath_train + path_train
        train_labels = np.array([[0,1]for i in range(nb_npath_training_point)]+[[1,0]for i in range(nb_path_training_point)])
        npath_test_mut = pre_process_data["npath_mutations"][nb_npath_training_point:]
        path_test_mut = pre_process_data["path_mutations"][nb_path_training_point:]
        all_lab_test_seq = npath_test_mut + path_test_mut
        test_labels = np.array([[0,1]for i in range(len(pre_process_data["npath_mutations"])-nb_npath_training_point)]+[[1,0]for i in range(len(pre_process_data["path_mutations"])-nb_path_training_point)])
        return {"all": all_lab_train_seq, "unlabeled": [], "labeled": all_lab_train_seq, "labels": train_labels, "test": all_lab_test_seq,
                "test_labels": test_labels, "npath_test": npath_test_mut, "path_test": path_test_mut,
                "npath_train": npath_train, "path_train": path_train}

def preprocess_get_label(gene):
    print("Gene: {}".format(gene))
    page1 = requests.get("https://www.alzforum.org/mutations/{}".format(gene))
    soup1 = BeautifulSoup(page1.content, 'html.parser')

    mutations = soup1.find_all('th', {"data-title": "Mutation"})
    mutations = [item.get_text() for item in mutations]

    pathogenicity = soup1.find_all('td', {"data-title": "Pathogenicity"})
    pathogenicity = [item.get_text().replace(' ', '').split(',') for item in pathogenicity]
    pathogenicity = [[disease.split(':') for disease in mutation] for mutation in pathogenicity]

    print("Number of mutations: {}".format(len(mutations)))

    valid_idx = []

    for idx, mutation in enumerate(mutations):
        if '+' in mutation or '-' in mutation or '.' in mutation or 'del' in mutation or '*' in mutation:
            continue
        elif '(' in mutation:
            mutations[idx] = mutation.split('(')[0]
            valid_idx.append(idx)
        else:
            valid_idx.append(idx)

    mutations = [mutations[i] for i in valid_idx]
    pathogenicity = [pathogenicity[i] for i in valid_idx]

    print("Number of substitutions: {}".format(len(mutations)))

    page2 = requests.get("https://www.uniprot.org/uniprot/?query={}&sort=score".format(gene))
    soup2 = BeautifulSoup(page2.content, 'html.parser')

    code = soup2.find("td", {"class": "entryID"}).get_text()

    page3 = requests.get("https://www.uniprot.org/uniprot/{}".format(code))
    soup3 = BeautifulSoup(page3.content, 'html.parser')

    isoforms = [isoform_code.get_text() for isoform_code in
                (isoforms.find('strong') for isoforms in soup3.findAll('div', {"class": "sequence-isoform"})) if
                isoform_code]

    print("Number of isoforms: {}".format(len(isoforms)))

    seqs = []

    for isoform_code in isoforms:
        page = requests.get("https://www.uniprot.org/blast/?about={}".format(isoform_code))
        soup = BeautifulSoup(page.content, 'html.parser')
        seq = soup.find(id="blastQuery").get_text()
        seqs.append(''.join(seq.split('\n')[1:-1]))

    nb_error = []

    for isoform in seqs:
        errors = 0
        for mutation in mutations:
            if int(mutation[1:-1]) > len(isoform) or isoform[int(mutation[1:-1]) - 1] != mutation[0]:
                errors += 1
        nb_error.append(errors)

    min_idx = 0
    min_error = nb_error[0]

    for i, error in enumerate(nb_error[1:]):
        if min_error > error:
            min_idx = i + 1
            min_error = error

    healthy_seq = seqs[min_idx]

    print("Isoform that fit the best the data: {}".format(isoforms[min_idx]))

    valid_idx = []

    for idx, mutation in enumerate(mutations):
        if int(mutation[1:-1]) > len(healthy_seq) or healthy_seq[int(mutation[1:-1]) - 1] != mutation[0]:
            continue
        else:
            valid_idx.append(idx)

    print("Number of valid substitutions for this isoform: {}".format(len(valid_idx)))

    mutations = [mutations[i] for i in valid_idx]

    mutation_index = [mutation[1:-1] for mutation in mutations]
    # print(mutation_index)

    pathogenicity = [pathogenicity[i] for i in valid_idx]

    diseases = {}
    occ_diseases = {}

    features_2 = {}
    features_2['Pathogenic'] = 0
    features_2['NotPathogenic'] = 1

    for mutation in pathogenicity:
        for disease in mutation:
            if disease[0] not in diseases:
                diseases[disease[0]] = len(diseases)
                occ_diseases[disease[0]] = 1
            else:
                occ_diseases[disease[0]] += 1

    init = False

    for disease in occ_diseases:
        if not init:
            max_rep_disease = disease
            init = True
        if occ_diseases[max_rep_disease] < occ_diseases[disease]:
            max_rep_disease = disease

    print("Most frequent disease: {}".format(max_rep_disease))

    return mutations, features_2, pathogenicity, max_rep_disease, healthy_seq

def get_label(mutations, features_2, pathogenicity, max_rep_disease, defocus_index, nb_train_path = -1, nb_npath_train = -1, most_frequent = True, nb_test = 5, all_diseases = False):

    labeled_mutations_2_features = []
    unlabeled_mutations_2_features = []
    all_mutations = []

    path_test_mut = []
    npath_test_mut = []

    features_freq = {}

    max_labels_2_features = np.zeros((len(mutations)-2*nb_test, len(features_2)))
    index_labelled_mut_2_features = []
    path_train = []
    npath_train = []
    test_labels_2_features = np.zeros((2 * nb_test, len(features_2)))
    test_mutation = []
    current_test_mut_index = 0
    count_path_test = 0
    count_npath_test = 0
    count_path_train = 0
    count_npath_train = 0
    rejected = 0
    # print(defocus_index)


    for i, mutation in enumerate(pathogenicity):
        # print(mutations[i][1:-1])
        # print(int(mutations[i][1:-1]) not in defocus_index)
        unlabelled = True
        test = False
        rejected_bool = False
        for disease in mutation:
            if disease[0] == max_rep_disease:
                unlabelled = False
                if disease[1] not in features_freq:
                    features_freq[disease[1]] = 0
                features_freq[disease[1]] += 1
                if count_npath_test<nb_test and disease[1] == 'NotPathogenic' and int(mutations[i][1:-1]) not in defocus_index:
                    test = True
                    test_mutation.append(mutations[i])
                    npath_test_mut.append(mutations[i])
                    test_labels_2_features[current_test_mut_index][features_2[disease[1]]] = 1
                    count_npath_test +=1
                    current_test_mut_index +=1
                elif count_path_test<nb_test and disease[1] == 'Pathogenic' and int(mutations[i][1:-1]) not in defocus_index:
                    test = True
                    test_mutation.append(mutations[i])
                    path_test_mut.append(mutations[i])
                    test_labels_2_features[current_test_mut_index][features_2[disease[1]]] = 1
                    count_path_test += 1
                    current_test_mut_index+=1
                else:
                    if disease[1] == 'UnclearPathogenicity':
                        unlabelled = True
                        # unlabeled_mutations_2_features.append(mutations[i])
                    elif int(mutations[i][1:-1]) not in defocus_index:

                        if embeed_feature(disease[1]) == 'Pathogenic' and (nb_train_path == -1 or count_path_train<nb_train_path):
                            labeled_mutations_2_features.append(mutations[i])
                            max_labels_2_features[i - current_test_mut_index][
                                features_2[embeed_feature(disease[1])]] = 1
                            index_labelled_mut_2_features.append(i - current_test_mut_index)
                            path_train.append(mutations[i])
                            count_path_train+=1
                        elif embeed_feature(disease[1]) == 'NotPathogenic' and (nb_npath_train == -1 or count_npath_train<nb_npath_train):
                            labeled_mutations_2_features.append(mutations[i])
                            max_labels_2_features[i - current_test_mut_index][
                                features_2[embeed_feature(disease[1])]] = 1
                            index_labelled_mut_2_features.append(i - current_test_mut_index)
                            npath_train.append(mutations[i])
                            count_npath_train+=1
                    else:
                        rejected_bool = True
                        rejected+=1
                break


        if unlabelled:
            unlabeled_mutations_2_features.append(mutations[i])
        if not test and not rejected_bool:
            all_mutations.append(mutations[i])
    # print("rejected:{}".format(rejected))

    # print("Number of labeled mutations: {}".format(len(labeled_mutations)))

    return all_mutations, unlabeled_mutations_2_features, labeled_mutations_2_features, max_labels_2_features[index_labelled_mut_2_features], test_mutation, test_labels_2_features, npath_test_mut, path_test_mut, npath_train, path_train


def get_label_old(mutations, features, features_2, pathogenicity, max_rep_disease, healthy_seq, defocus_index, nb_train_path = -1, nb_npath_train = -1, most_frequent = True, nb_test = 5, all_diseases = False):

    labeled_mutations = []
    labeled_mutations_2_features = []
    unlabeled_mutations = []
    unlabeled_mutations_2_features = []
    all_mutations = []

    path_test_mut = []
    npath_test_mut = []

    features_freq = {}

    max_labels = np.zeros((len(mutations)-2*nb_test, len(features)))
    max_labels_2_features = np.zeros((len(mutations)-2*nb_test, len(features_2)))
    index_labelled_mut = []
    index_labelled_mut_2_features = []
    path_train = []
    npath_train = []
    test_labels = np.zeros((2*nb_test, len(features)))
    test_labels_2_features = np.zeros((2 * nb_test, len(features_2)))
    test_mutation = []
    current_test_mut_index = 0
    count_path_test = 0
    count_npath_test = 0
    count_path_train = 0
    count_npath_train = 0
    rejected = 0
    # print(defocus_index)


    for i, mutation in enumerate(pathogenicity):
        # print(mutations[i][1:-1])
        # print(int(mutations[i][1:-1]) not in defocus_index)
        unlabelled = True
        test = False
        rejected_bool = False
        for disease in mutation:
            if disease[0] == max_rep_disease:
                unlabelled = False
                if disease[1] not in features_freq:
                    features_freq[disease[1]] = 0
                features_freq[disease[1]] += 1
                if count_npath_test<nb_test and disease[1] == 'NotPathogenic' and int(mutations[i][1:-1]) not in defocus_index:
                    test = True
                    test_mutation.append(mutations[i])
                    npath_test_mut.append(mutations[i])
                    test_labels[current_test_mut_index][features[disease[1]]] = 1
                    test_labels_2_features[current_test_mut_index][features_2[disease[1]]] = 1
                    count_npath_test +=1
                    current_test_mut_index +=1
                elif count_path_test<nb_test and disease[1] == 'Pathogenic' and int(mutations[i][1:-1]) not in defocus_index:
                    test = True
                    test_mutation.append(mutations[i])
                    path_test_mut.append(mutations[i])
                    test_labels[current_test_mut_index][features[disease[1]]] = 1
                    test_labels_2_features[current_test_mut_index][features_2[disease[1]]] = 1
                    count_path_test += 1
                    current_test_mut_index+=1
                else:
                    max_labels[i-current_test_mut_index][features[disease[1]]] = 1
                    labeled_mutations.append(mutations[i])
                    index_labelled_mut.append(i-current_test_mut_index)
                    if disease[1] == 'UnclearPathogenicity':
                        unlabelled = True
                        # unlabeled_mutations_2_features.append(mutations[i])
                    elif int(mutations[i][1:-1]) not in defocus_index:

                        if embeed_feature(disease[1]) == 'Pathogenic' and (nb_train_path == -1 or count_path_train<nb_train_path):
                            labeled_mutations_2_features.append(mutations[i])
                            max_labels_2_features[i - current_test_mut_index][
                                features_2[embeed_feature(disease[1])]] = 1
                            index_labelled_mut_2_features.append(i - current_test_mut_index)
                            path_train.append(mutations[i])
                            count_path_train+=1
                        elif embeed_feature(disease[1]) == 'NotPathogenic' and (nb_npath_train == -1 or count_npath_train<nb_npath_train):
                            labeled_mutations_2_features.append(mutations[i])
                            max_labels_2_features[i - current_test_mut_index][
                                features_2[embeed_feature(disease[1])]] = 1
                            index_labelled_mut_2_features.append(i - current_test_mut_index)
                            npath_train.append(mutations[i])
                            count_npath_train+=1
                    else:
                        rejected_bool = True
                        rejected+=1
                break


        if unlabelled:
            unlabeled_mutations.append(mutations[i])
            unlabeled_mutations_2_features.append(mutations[i])
        if not test and not rejected_bool:
            all_mutations.append(mutations[i])
    # print("rejected:{}".format(rejected))

    # print("Number of labeled mutations: {}".format(len(labeled_mutations)))

    return all_mutations, unlabeled_mutations, labeled_mutations, max_labels[index_labelled_mut], unlabeled_mutations_2_features, labeled_mutations_2_features, max_labels_2_features[index_labelled_mut_2_features], healthy_seq, test_mutation, test_labels, test_labels_2_features, npath_test_mut, path_test_mut, npath_train, path_train, features, features_2

# def get_seq(mutations_list, healthy_seq, focus_index = None, defocus_index = None):
#     seq_list = []
#     # labels_to_keep = []
#     for i, mutation in enumerate(mutations_list):
#         # original = mutation[0] #to check
#         mut = mutation[-1]
#         pos = int(mutation[1:-1])
#         # if defocus_index is not None:
#         #     if pos in defocus_index:
#         #         continue
#         #     else:
#         #         labels_to_keep.append(i)
#         # labels_to_keep.append(i)
#         if pos == 1:
#             new_seq = mut + healthy_seq[1:]
#         elif pos == len(healthy_seq):
#             new_seq = healthy_seq[:-1] + mut
#         else:
#             new_seq = healthy_seq[:pos-1]+mut+healthy_seq[pos:]
#         if focus_index is not None:
#             new_seq = [new_seq[ix] for ix in focus_index]
#         seq_list.append(new_seq)
#     # if defocus_index is None:
#     return seq_list
#     # else:
#     #     return seq_list, labels_to_keep



def KLD(mu, log_sigma, prior_mu, prior_log_sigma):
    return prior_log_sigma - log_sigma + 0.5 * (math.exp(2. * log_sigma) + (mu - prior_mu) ** 2) * math.exp(
        -2. * prior_log_sigma) - 0.5

def JSD(mu1, log_sigma1, mu2, log_sigma2):
    return 0.5*(KLD(mu1, log_sigma1, mu2, log_sigma2)+KLD(mu2, log_sigma2, mu1, log_sigma1))


def get_seq(mutations_list, healthy_seq, focus_index = None, defocus_index = None, data_format = "manual"):
    seq_list = []
    # labels_to_keep = []
    for i, mutation in enumerate(mutations_list):
        # original = mutation[0] #to check
        mut = mutation[-1]
        pos = int(mutation[1:-1])
        if data_format == "manual" and defocus_index is not None:
            if pos in defocus_index:
                continue
        if pos == 1:
            new_seq = mut + healthy_seq[1:]
        elif pos == len(healthy_seq):
            new_seq = healthy_seq[:-1] + mut
        else:
            new_seq = healthy_seq[:pos-1]+mut+healthy_seq[pos:]
        if focus_index is not None:
            new_seq = [new_seq[ix] for ix in focus_index]
        seq_list.append(new_seq)
    # if defocus_index is None:
    return seq_list
    # else:
    #     return seq_list, labels_to_keep

def get_one_hot(seq_list):
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    aa_dict = {}
    for i, aa in enumerate(alphabet):
        aa_dict[aa] = i
    one_hot = np.zeros((len(seq_list), len(seq_list[0]), len(alphabet)))
    for i, seq in enumerate(seq_list):
        for j, letter in enumerate(seq):
            if letter in aa_dict:
                k = aa_dict[letter]
                one_hot[i, j, k] = 1.0
    return one_hot

def create_labels(batch, nb_label):
    labels = torch.zeros(batch.shape[0]*nb_label, nb_label)
    for i in range(nb_label):
        for j in range(batch.shape[0]):
            labels[j+batch.shape[0]*i][i] = 1
    return labels

def log_standard_categorical(p):
    """
    Calculates the cross entropy between a (one-hot) categorical vector
    and a standard (uniform) categorical distribution.
    :param p: one-hot categorical distribution
    :return: H(p, u)
    """
    # Uniform prior over y
    prior = F.softmax(torch.ones_like(p), dim=1)
    prior.requires_grad = False

    cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)

    return cross_entropy

def pred_from_onehot(onehot, healthy_seq_one_hot, model, alphabet_size, sequence_len, semi_supervised = False, N_pred_iterations=500, minibatch_size=2000, filename_prefix="", offset=0):

    prediction_matrix = np.zeros((onehot.shape[0], N_pred_iterations))

    batch_order = np.arange(onehot.shape[0])

    healthy_pred = []

    for i in range(N_pred_iterations):

        np.random.shuffle(batch_order)

        for j in range(0, onehot.shape[0], minibatch_size):
            #                 print(j)

            batch_index = batch_order[j:j + minibatch_size]

            #                 print(self.mutant_sequences_one_hot[batch_index].shape)

            batch = onehot[batch_index]

            batch = batch.reshape(-1, alphabet_size * sequence_len)

            # print(batch.shape)

            batch = torch.Tensor(batch).to(device)

            if semi_supervised:
                mu, logsigma, px_zy, logpx_zy, qy_x, logqy_x, y = model(batch)[:7]
                batch_preds = unlabelled_mut_loss_no_mean(logpx_zy, mu, logsigma, qy_x, logqy_x, y)
                #                 print(batch_preds)

                mu, logsigma, px_zy, logpx_zy, qy_x, logqy_x, y = model((torch.Tensor(healthy_seq_one_hot)).to(device))[:7]

                healthy_pred.append(unlabelled_mut_loss_no_mean(logpx_zy, mu, logsigma, qy_x, logqy_x, y).item())

                batch_preds_numpy = batch_preds.cpu().numpy()

            else:
                mu, logsigma, _, logpx_z, z = model(batch)[0:5]

                batch_preds = ELBO_no_mean(logpx_z, mu, logsigma, z, 1.0)
                #                 print(batch_preds)



                mu, logsigma, _, logpx_z, z = model((torch.Tensor(healthy_seq_one_hot)).to(device))[0:5]

                healthy_pred.append(ELBO_no_mean(logpx_z, mu, logsigma, z, 1.0).item())

                batch_preds_numpy = batch_preds.cpu().numpy()

            #                 print(batch_index.shape)

            for k, idx_batch in enumerate(batch_index.tolist()):
                prediction_matrix[idx_batch][i] = batch_preds_numpy[k]



    # Then take the mean of all my elbo samples
    mean_elbos = np.mean(prediction_matrix, axis=1)
    # print(mean_elbos.shape)
    # print(mean_elbos[0])

    wt_elbo = np.mean(healthy_pred)
    # print(wt_elbo)

    delta_elbos = mean_elbos - wt_elbo
    # print(delta_elbos.shape)
    # print(delta_elbos[0])

    return delta_elbos

class DataHelper:
    def __init__(self,
        dataset,
        theta,
        alignment_file="",
        custom_dataset = False,
        focus_seq_name="",
        calc_weights=True,
        working_dir=".",
        load_all_sequences=True,
        alphabet_type="protein"):

        """
        Class to load and organize alignment data.
        This function also helps makes predictions about mutations.

        Parameters
        --------------
        dataset: preloaded dataset names
                    We have found it easiest to organize datasets in this
                    way and use the self.configure_datasets() func
        alignment_file: Name of the alignment file located in the "datasets"
                            folder. Not needed if dataset pre-entered
        focus_seq_name: Name of the sequence in the alignment
                            Defaults to the first sequence in the alignment
        calc_weights: (bool) Calculate sequence weights
                        Default True, but not necessary if just loading weights
                            and doing mutation effect prediction
        working_dir: location of "params", "logs", "embeddings", and "datasets"
                        folders
        theta: Sequence weighting hyperparameter
                Generally: Prokaryotic and eukaryotic families =  0.2
                            Viruses = 0.01
        load_all_sequences:
        alphabet_type: Alphabet type of associated dataset.
                            Options are DNA, RNA, protein, allelic

        Returns
        ------------
        None
        """

        # np.random.seed(42)
        self.dataset = dataset
        self.alignment_file = alignment_file
        self.focus_seq_name = focus_seq_name
        self.working_dir = working_dir
        self.calc_weights = calc_weights
        self.alphabet_type = alphabet_type
        
        if theta == 0:
            self.calc_weigths = False

        # Initalize the elbo of the wt to None
        #   will be useful if eventually doing mutation effect prediction
        self.wt_elbo = None

        # Alignment processing parameters
        self.theta = theta

        # If I am running tests with the model, I don't need all the
        #    sequences loaded
        self.load_all_sequences = load_all_sequences

        # Load necessary information for preloaded datasets
        if custom_dataset:
            self.alignment_file = dataset

        elif self.dataset != "":
            self.configure_datasets()

        # Load up the alphabet type to use, whether that be DNA, RNA, or protein
        if self.alphabet_type == "protein":
            self.alphabet = "ACDEFGHIKLMNPQRSTVWY"
            self.reorder_alphabet = "DEKRHNQSTPGAVILMCFYW"
        elif self.alphabet_type == "RNA":
            self.alphabet = "ACGU"
            self.reorder_alphabet = "ACGU"
        elif self.alphabet_type == "DNA":
            self.alphabet = "ACGT"
            self.reorder_alphabet = "ACGT"
        elif self.alphabet_type == "allelic":
            self.alphabet = "012"
            self.reorder_alphabet = "012"

        #then generate the experimental data
        self.gen_basic_alignment()

        if self.load_all_sequences:
            self.gen_full_alignment()

    def configure_datasets(self):

        if self.dataset == "BLAT_ECOLX":
            self.alignment_file = self.working_dir+"/datasets/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m"
#             self.theta = 0.2

        elif self.dataset == "PABP_YEAST":
            self.alignment_file = self.working_dir+"/datasets/PABP_YEAST_hmmerbit_plmc_n5_m30_f50_t0.2_r115-210_id100_b48.a2m"
#             self.theta = 0.2

        elif self.dataset == "DLG4_RAT":
            self.alignment_file = self.working_dir+"/datasets/DLG4_RAT_hmmerbit_plmc_n5_m30_f50_t0.2_r300-400_id100_b50.a2m"
#             self.theta = 0.2

        elif self.dataset == "BG505":
            self.alignment_file = self.working_dir+"/datasets/BG505_env_1_b0.5.a2m"
#             self.theta = 0.2
            
        elif self.dataset == "BF520":
            self.alignment_file = self.working_dir+"/datasets/BF520_env_1_b0.5.a2m"
#             self.theta = 0.01

        elif self.dataset == "trna":
            self.alignment_file = self.working_dir+"/datasets/RF00005_CCU.fasta"
            self.alphabet_type = "RNA"
#             self.theta = 0.2


    def one_hot_3D(self, s):
        """ Transform sequence string into one-hot aa vector"""
        # One-hot encode as row vector
        x = np.zeros((len(s), len(self.alphabet)))
        for i, letter in enumerate(s):
            if letter in self.aa_dict:
                x[i , self.aa_dict[letter]] = 1
        return x

    def gen_basic_alignment(self):
        """ Read training alignment and store basics in class instance """
        # Make a dictionary that goes from aa to a number for one-hot
        self.aa_dict = {}
        for i,aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i

        # Do the inverse as well
        self.num_to_aa = {i:aa for aa,i in self.aa_dict.items()}

        ix = np.array([self.alphabet.find(s) for s in self.reorder_alphabet])

        # Read alignment
        self.seq_name_to_sequence = defaultdict(str)
        self.seq_names = []

        name = ""
        INPUT = open(self.alignment_file, "r")
        for i, line in enumerate(INPUT):
            line = line.rstrip()
            if line.startswith(">"):
                name = line
                self.seq_names.append(name)
            else:
                self.seq_name_to_sequence[name] += line
        INPUT.close()

        # If we don"t have a focus sequence, pick the one that
        #   we used to generate the alignment
        if self.focus_seq_name == "":
            self.focus_seq_name = self.seq_names[0]

        # Select focus columns
        #  These columns are the uppercase residues of the .a2m file
        self.focus_seq = self.seq_name_to_sequence[self.focus_seq_name]
        self.focus_index = [ix for ix, s in enumerate(self.focus_seq) if s == s.upper()]
        self.defocus_index = [ix for ix, s in enumerate(self.focus_seq) if s != s.upper()]
        self.focus_cols = [ix for ix, s in enumerate(self.focus_seq) if s == s.upper()]
        self.focus_seq_trimmed = [self.focus_seq[ix] for ix in self.focus_cols]
        self.seq_len = len(self.focus_cols)
        self.alphabet_size = len(self.alphabet)

        # We also expect the focus sequence to be formatted as:
        # >[NAME]/[start]-[end]
        focus_loc = self.focus_seq_name.split("/")[-1]
        start,stop = focus_loc.split("-")
        self.focus_start_loc = int(start)
        self.focus_stop_loc = int(stop)
        self.uniprot_focus_cols_list \
            = [idx_col+int(start) for idx_col in self.focus_cols]
        self.uniprot_focus_col_to_wt_aa_dict \
            = {idx_col+int(start):self.focus_seq[idx_col] for idx_col in self.focus_cols}
        self.uniprot_focus_col_to_focus_idx \
            = {idx_col+int(start):idx_col for idx_col in self.focus_cols}


    def gen_full_alignment(self):

        # Get only the focus columns
        for seq_name,sequence in self.seq_name_to_sequence.items():
            # Replace periods with dashes (the uppercase equivalent)
            sequence = sequence.replace(".","-")

            #then get only the focus columns
            self.seq_name_to_sequence[seq_name] = [sequence[ix].upper() for ix in self.focus_cols]

        # Remove sequences that have bad characters
        alphabet_set = set(list(self.alphabet))
        seq_names_to_remove = []
        for seq_name,sequence in self.seq_name_to_sequence.items():
            for letter in sequence:
                if letter not in alphabet_set and letter != "-":
                    seq_names_to_remove.append(seq_name)

        seq_names_to_remove = list(set(seq_names_to_remove))
        for seq_name in seq_names_to_remove:
            del self.seq_name_to_sequence[seq_name]

        # Encode the sequences
        print ("Encoding sequences")
        self.x_train = np.zeros((len(self.seq_name_to_sequence.keys()),len(self.focus_cols),len(self.alphabet)))
        self.x_train_name_list = []
        for i,seq_name in enumerate(self.seq_name_to_sequence.keys()):
            sequence = self.seq_name_to_sequence[seq_name]
            self.x_train_name_list.append(seq_name)
            for j,letter in enumerate(sequence):
                if letter in self.aa_dict:
                    k = self.aa_dict[letter]
                    self.x_train[i,j,k] = 1.0
                    
        #Very fast weight computation
        
        self.seqlen = self.x_train.shape[1]
        self.datasize = self.x_train.shape[0]
        
        if self.calc_weights and self.theta>0:
            print("effective weigths")
            weights = []
            seq_batch = 1000
            x_train_flat = self.x_train.reshape(self.x_train.shape[0], -1)
            nb_seq = x_train_flat.shape[0]
            nb_iter = int(nb_seq/seq_batch)
            rest = nb_seq%seq_batch
            xtfs_t = torch.Tensor(x_train_flat).float().to(device)
            for i in range(nb_iter):
                weights.append(1.0 / (((torch.div(torch.mm(xtfs_t[i*seq_batch:(i+1)*seq_batch], xtfs_t.transpose(0,1)), xtfs_t[i*seq_batch:(i+1)*seq_batch].sum(1).unsqueeze(1))) > (1 - self.theta)).sum(1).float()))
            weights.append(1.0 / (((torch.div(torch.mm(xtfs_t[-rest:], xtfs_t.transpose(0,1)), xtfs_t[-rest:].sum(1).unsqueeze(1))) > (1 - self.theta)).sum(1).float()))
            weights_tensor = torch.cat(weights)
            self.weights = weights_tensor
    #         self.weights = weights_tensor.cpu().numpy()
            self.Neff = weights_tensor.sum()
#             print(self.Neff)
        else:
#             # If not using weights, use an isotropic weight matrix
            self.weights = np.ones(self.x_train.shape[0])
            self.Neff = self.x_train.shape[0]
    
        print ("Neff =",str(self.Neff))
        print ("Data Shape =",self.x_train.shape)


#         # Fast sequence weights with Theano
#         if self.calc_weights:
#             print ("Computing sequence weights")
#             # Numpy version
#             import scipy
#             from scipy.spatial.distance import pdist, squareform
#             x_train_flat = self.x_train.reshape(self.x_train.shape[0], -1)
#             print(x_train_flat.shape)
# #             self.weights = 1.0 / np.sum(squareform(pdist(x_train_flat[:10000], metric="hamming")) < self.theta, axis=0)
#             self.weights = 1.0 / np.sum(squareform(pdist(x_train_flat, metric="hamming")) < self.theta, axis=0)
#             #
#             # Theano weights
#             # X = T.tensor3("x")
#             # cutoff = T.scalar("theta")
#             # X_flat = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
#             # N_list, updates = theano.map(lambda x: 1.0 / T.sum(T.dot(X_flat, x) / T.dot(x, x) > 1 - cutoff), X_flat)
#             # weightfun = theano.function(inputs=[X, cutoff], outputs=[N_list],allow_input_downcast=True)
#             #
#             # self.weights = weightfun(self.x_train, self.theta)[0]

#         else:
#             # If not using weights, use an isotropic weight matrix
#             self.weights = np.ones(self.x_train.shape[0])

#         self.Neff = np.sum(self.weights)

#         print ("Neff =",str(self.Neff))
#         print ("Data Shape =",self.x_train.shape)

        


    def delta_elbo(self, model, mutant_tuple_list, N_pred_iterations=10):

        for pos,wt_aa,mut_aa in mutant_tuple_list:
            if pos not in self.uniprot_focus_col_to_wt_aa_dict \
                or self.uniprot_focus_col_to_wt_aa_dict[pos] != wt_aa:
                print ("Not a valid mutant!",pos,wt_aa,mut_aa)
                return None

        mut_seq = self.focus_seq_trimmed[:]
        for pos,wt_aa,mut_aa in mutant_tuple_list:
            mut_seq[self.uniprot_focus_col_to_focus_idx[pos]] = mut_aa


        if self.wt_elbo == None:
            mutant_sequences = [self.focus_seq_trimmed, mut_seq]
        else:
            mutant_sequences = [mut_seq]

        # Then make the one hot sequence
        mutant_sequences_one_hot = np.zeros(\
            (len(mutant_sequences),len(self.focus_cols),len(self.alphabet)))

        for i,sequence in enumerate(mutant_sequences):
            for j,letter in enumerate(sequence):
                k = self.aa_dict[letter]
                mutant_sequences_one_hot[i,j,k] = 1.0

        prediction_matrix = np.zeros((mutant_sequences_one_hot.shape[0],N_pred_iterations))
        idx_batch = np.arange(mutant_sequences_one_hot.shape[0])
        for i in range(N_pred_iterations):

            batch_preds, _, _ = model.all_likelihood_components(mutant_sequences_one_hot)

            prediction_matrix[:,i] = batch_preds

        # Then take the mean of all my elbo samples
        mean_elbos = np.mean(prediction_matrix, axis=1).flatten().tolist()

        if self.wt_elbo == None:
            self.wt_elbo = mean_elbos.pop(0)

        return mean_elbos[0] - self.wt_elbo

    def single_mutant_matrix(self, model, N_pred_iterations=10, \
            minibatch_size=2000, filename_prefix=""):

        """ Predict the delta elbo for all single mutants """

        # Get the start and end index from the sequence name
        start_idx, end_idx = self.focus_seq_name.split("/")[-1].split("-")
        start_idx = int(start_idx)

        wt_pos_focus_idx_tuple_list = []
        focus_seq_index = 0
        focus_seq_list = []
        for i,letter in enumerate(self.focus_seq):
            if letter == letter.upper():
                wt_pos_focus_idx_tuple_list.append((letter,start_idx+i,focus_seq_index))
                focus_seq_index += 1

        self.mutant_sequences = ["".join(self.focus_seq_trimmed)]
        self.mutant_sequences_descriptor = ["wt"]
        for wt,pos,idx_focus in wt_pos_focus_idx_tuple_list:
            for mut in self.alphabet:
                if wt != mut:
                    # Make a descriptor
                    descriptor = wt+str(pos)+mut

                    # Hard copy the sequence
                    focus_seq_copy = list(self.focus_seq_trimmed)[:]

                    # Mutate
                    focus_seq_copy[idx_focus] = mut

                    # Add to the list
                    self.mutant_sequences.append("".join(focus_seq_copy))
                    self.mutant_sequences_descriptor.append(descriptor)

        # Then make the one hot sequence
        self.mutant_sequences_one_hot = np.zeros(\
            (len(self.mutant_sequences),len(self.focus_cols),len(self.alphabet)))

        for i,sequence in enumerate(self.mutant_sequences):
            for j,letter in enumerate(sequence):
                k = self.aa_dict[letter]
                self.mutant_sequences_one_hot[i,j,k] = 1.0

        self.prediction_matrix = np.zeros((self.mutant_sequences_one_hot.shape[0],N_pred_iterations))

        batch_order = np.arange(self.mutant_sequences_one_hot.shape[0])

        for i in range(N_pred_iterations):
            np.random.shuffle(batch_order)

            for j in range(0,self.mutant_sequences_one_hot.shape[0],minibatch_size):

                batch_index = batch_order[j:j+minibatch_size]
                batch_preds, _, _ = model.all_likelihood_components(self.mutant_sequences_one_hot[batch_index])

                for k,idx_batch in enumerate(batch_index.tolist()):
                    self.prediction_matrix[idx_batch][i]= batch_preds[k]

        # Then take the mean of all my elbo samples
        self.mean_elbos = np.mean(self.prediction_matrix, axis=1).flatten().tolist()

        self.wt_elbo = self.mean_elbos.pop(0)
        self.mutant_sequences_descriptor.pop(0)

        self.delta_elbos = np.asarray(self.mean_elbos) - self.wt_elbo

        if filename_prefix == "":
            return self.mutant_sequences_descriptor, self.delta_elbos

        else:
            OUTPUT = open(filename_prefix+"_samples-"+str(N_pred_iterations)\
                +"_elbo_predictions.csv", "w")

            for i,descriptor in enumerate(self.mutant_sequences_descriptor):
                OUTPUT.write(descriptor+";"+str(self.mean_elbos[i])+"\n")

            OUTPUT.close()


    def custom_mutant_matrix(self, input_filename, model, N_pred_iterations=10, \
            minibatch_size=2000, filename_prefix="", offset=0):

        """ Predict the delta elbo for a custom mutation filename
        """
        # Get the start and end index from the sequence name
        start_idx, end_idx = self.focus_seq_name.split("/")[-1].split("-")
        start_idx = int(start_idx)

        wt_pos_focus_idx_tuple_list = []
        focus_seq_index = 0
        focus_seq_list = []
        mutant_to_letter_pos_idx_focus_list = {}

        # find all possible valid mutations that can be run with this alignment
        for i,letter in enumerate(self.focus_seq):
            if letter == letter.upper():
                for mut in self.alphabet:
                    pos = start_idx+i
                    if letter != mut:
                        mutant = letter+str(pos)+mut
                        mutant_to_letter_pos_idx_focus_list[mutant] = [letter,start_idx+i,focus_seq_index]
                focus_seq_index += 1

        self.mutant_sequences = ["".join(self.focus_seq_trimmed)]
        self.mutant_sequences_descriptor = ["wt"]

        # run through the input file
        INPUT = open(self.working_dir+"/"+input_filename, "r")
        for i,line in enumerate(INPUT):
            line = line.rstrip()
            if i >= 1:
                line_list = line.split(",")
                # generate the list of mutants
                mutant_list = line_list[0].split(":")
                valid_mutant = True

                # if any of the mutants in this list aren"t in the focus sequence,
                #    I cannot make a prediction
                for mutant in mutant_list:
                    if mutant not in mutant_to_letter_pos_idx_focus_list:
                        valid_mutant = False

                # If it is a valid mutant, add it to my list to make preditions
                if valid_mutant:
                    focus_seq_copy = list(self.focus_seq_trimmed)[:]

                    for mutant in mutant_list:
                        wt_aa,pos,idx_focus = mutant_to_letter_pos_idx_focus_list[mutant]
                        mut_aa = mutant[-1]
                        focus_seq_copy[idx_focus] = mut_aa

                    self.mutant_sequences.append("".join(focus_seq_copy))
                    self.mutant_sequences_descriptor.append(":".join(mutant_list))

        INPUT.close()

        # Then make the one hot sequence
        self.mutant_sequences_one_hot = np.zeros(\
            (len(self.mutant_sequences),len(self.focus_cols),len(self.alphabet)))

        for i,sequence in enumerate(self.mutant_sequences):
            for j,letter in enumerate(sequence):
                k = self.aa_dict[letter]
                self.mutant_sequences_one_hot[i,j,k] = 1.0

        self.prediction_matrix = np.zeros((self.mutant_sequences_one_hot.shape[0],N_pred_iterations))

        batch_order = np.arange(self.mutant_sequences_one_hot.shape[0])

        for i in range(N_pred_iterations):
            np.random.shuffle(batch_order)

            for j in range(0,self.mutant_sequences_one_hot.shape[0],minibatch_size):

                batch_index = batch_order[j:j+minibatch_size]
                batch_preds, _, _ = model.all_likelihood_components(self.mutant_sequences_one_hot[batch_index])

                for k,idx_batch in enumerate(batch_index.tolist()):
                    self.prediction_matrix[idx_batch][i]= batch_preds[k]

        # Then take the mean of all my elbo samples
        self.mean_elbos = np.mean(self.prediction_matrix, axis=1).flatten().tolist()

        self.wt_elbo = self.mean_elbos.pop(0)
        self.mutant_sequences_descriptor.pop(0)

        self.delta_elbos = np.asarray(self.mean_elbos) - self.wt_elbo

        if filename_prefix == "":
            return self.mutant_sequences_descriptor, self.delta_elbos

        else:

            OUTPUT = open(filename_prefix+"_samples-"+str(N_pred_iterations)\
                +"_elbo_predictions.csv", "w")

            for i,descriptor in enumerate(self.mutant_sequences_descriptor):
                OUTPUT.write(descriptor+";"+str(self.delta_elbos[i])+"\n")

            OUTPUT.close()

    def custom_mutant_matrix_pytorch(self, input_filename, model, N_pred_iterations=10, \
            minibatch_size=2000, filename_prefix="", offset=0):

        """ Predict the delta elbo for a custom mutation filename
        """
        # Get the start and end index from the sequence name
        start_idx, end_idx = self.focus_seq_name.split("/")[-1].split("-")
        start_idx = int(start_idx)

        wt_pos_focus_idx_tuple_list = []
        focus_seq_index = 0
        focus_seq_list = []
        mutant_to_letter_pos_idx_focus_list = {}

        # find all possible valid mutations that can be run with this alignment
        for i,letter in enumerate(self.focus_seq):
            if letter == letter.upper():
                for mut in self.alphabet:
                    pos = start_idx+i
                    if letter != mut:
                        mutant = letter+str(pos)+mut
                        mutant_to_letter_pos_idx_focus_list[mutant] = [letter,start_idx+i,focus_seq_index]
                focus_seq_index += 1

        self.mutant_sequences = ["".join(self.focus_seq_trimmed)]
        self.mutant_sequences_descriptor = ["wt"]

        # run through the input file
        INPUT = open(self.working_dir+"/"+input_filename, "r")
        for i,line in enumerate(INPUT):
            line = line.rstrip()
            if i >= 1:
                line_list = line.split(",")
                # generate the list of mutants
                mutant_list = line_list[0].split(":")
                valid_mutant = True

                # if any of the mutants in this list aren"t in the focus sequence,
                #    I cannot make a prediction
                for mutant in mutant_list:
                    if mutant not in mutant_to_letter_pos_idx_focus_list:
                        valid_mutant = False

                # If it is a valid mutant, add it to my list to make preditions
                if valid_mutant:
                    focus_seq_copy = list(self.focus_seq_trimmed)[:]

                    for mutant in mutant_list:
                        wt_aa,pos,idx_focus = mutant_to_letter_pos_idx_focus_list[mutant]
                        mut_aa = mutant[-1]
                        focus_seq_copy[idx_focus] = mut_aa

                    self.mutant_sequences.append("".join(focus_seq_copy))
                    self.mutant_sequences_descriptor.append(":".join(mutant_list))

        INPUT.close()

        # Then make the one hot sequence
        self.mutant_sequences_one_hot = np.zeros(\
            (len(self.mutant_sequences),len(self.focus_cols),len(self.alphabet)))

        for i,sequence in enumerate(self.mutant_sequences):
            for j,letter in enumerate(sequence):
                k = self.aa_dict[letter]
                self.mutant_sequences_one_hot[i,j,k] = 1.0

        self.prediction_matrix = np.zeros((self.mutant_sequences_one_hot.shape[0],N_pred_iterations))

        batch_order = np.arange(self.mutant_sequences_one_hot.shape[0])

        for i in range(N_pred_iterations):
#             print("i" + str(i))
            np.random.shuffle(batch_order)

            for j in range(0,self.mutant_sequences_one_hot.shape[0],minibatch_size):
#                 print(j)

                batch_index = batch_order[j:j+minibatch_size]

#                 print(self.mutant_sequences_one_hot[batch_index].shape)

                batch = self.mutant_sequences_one_hot[batch_index]

                batch = batch.reshape(-1, self.alphabet_size*self.seq_len)

                # print(batch.shape)

                batch = torch.Tensor(batch).to(device)

                mu, logsigma, _, logpx_z, _, _, _, _, _, _, _, _, _, _ ,_,_,_,_,_,_,_,_= model(batch)

                batch_preds = ELBO_no_mean(logpx_z, mu, logsigma, 1.0)
#                 print(batch_preds)
                
                batch_preds_numpy = batch_preds.cpu().numpy()
                
#                 print(batch_index.shape)

                for k,idx_batch in enumerate(batch_index.tolist()):
                    self.prediction_matrix[idx_batch][i]= batch_preds_numpy[k]

        # Then take the mean of all my elbo samples
        self.mean_elbos = np.mean(self.prediction_matrix, axis=1).flatten().tolist()

        self.wt_elbo = self.mean_elbos.pop(0)
        self.mutant_sequences_descriptor.pop(0)

        self.delta_elbos = np.asarray(self.mean_elbos) - self.wt_elbo

        if filename_prefix == "":
            return self.mutant_sequences_descriptor, self.delta_elbos

        else:

            OUTPUT = open(filename_prefix+"_samples-"+str(N_pred_iterations)\
                +"_elbo_predictions.csv", "w")

            for i,descriptor in enumerate(self.mutant_sequences_descriptor):
                OUTPUT.write(descriptor+";"+str(self.delta_elbos[i])+"\n")

            OUTPUT.close()

            return self.mutant_sequences_descriptor, self.delta_elbos


    def pred_from_onehot(self, onehot, healthy_seq_one_hot, model, N_pred_iterations=10, minibatch_size=2000, filename_prefix="", offset=0):

        prediction_matrix = np.zeros((onehot.shape[0], N_pred_iterations))

        batch_order = np.arange(onehot.shape[0])

        healthy_pred = []

        for i in range(N_pred_iterations):

            np.random.shuffle(batch_order)

            for j in range(0, onehot.shape[0], minibatch_size):
                #                 print(j)

                batch_index = batch_order[j:j + minibatch_size]

                #                 print(self.mutant_sequences_one_hot[batch_index].shape)

                batch = onehot[batch_index]

                batch = batch.reshape(-1, self.alphabet_size * self.seq_len)

                # print(batch.shape)

                batch = torch.Tensor(batch).to(device)

                mu, logsigma, _, logpx_z, z = model(batch)[0:5]

                batch_preds = ELBO_no_mean(logpx_z, mu, logsigma, z, 1.0)
                #                 print(batch_preds)

                mu, logsigma, _, logpx_z, z = model(healthy_seq_one_hot)[0:5]

                healthy_pred.append(ELBO_no_mean(logpx_z, mu, logsigma, z, 1.0).item())

                batch_preds_numpy = batch_preds.cpu().numpy()

                #                 print(batch_index.shape)

                for k, idx_batch in enumerate(batch_index.tolist()):
                    prediction_matrix[idx_batch][i] = batch_preds_numpy[k]



        # Then take the mean of all my elbo samples
        mean_elbos = np.mean(prediction_matrix, axis=1)
        # print(mean_elbos.shape)
        # print(mean_elbos[0])

        wt_elbo = np.mean(healthy_pred)
        # print(wt_elbo)

        delta_elbos = mean_elbos - wt_elbo
        # print(delta_elbos.shape)
        # print(delta_elbos[0])

        return delta_elbos



    def get_pattern_activations(self, model, update_num, filename_prefix="",
                        verbose=False, minibatch_size=2000):

        activations_filename = self.working_dir+"/embeddings/"+filename_prefix+"_pattern_activations.csv"

        OUTPUT = open(activations_filename, "w")

        batch_order = np.arange(len(self.x_train_name_list))

        for i in range(0,len(self.x_train_name_list),minibatch_size):
            batch_index = batch_order[i:i+minibatch_size]
            one_hot_seqs = self.x_train[batch_index]
            batch_activation = model.get_pattern_activations(one_hot_seqs)

            for j,idx in enumerate(batch_index.tolist()):
                sample_activation = [str(val) for val in batch_activation[j].tolist()]
                sample_name = self.x_train_name_list[idx]
                out_line = [str(update_num),sample_name]+sample_activation
                if verbose:
                    print ("\t".join(out_line))
                OUTPUT.write(",".join(out_line)+"\n")

        OUTPUT.close()


    def get_embeddings(self, model, update_num, filename_prefix="",
                        verbose=False, minibatch_size=2000):
        """ Save the latent variables from all the sequences in the alignment """
        embedding_filename = self.working_dir+"/embeddings/"+filename_prefix+"_seq_embeddings.csv"

        # Append embeddings to file if it has already been created
        #   This is useful if you want to see the embeddings evolve over time
        if os.path.isfile(embedding_filename):
            OUTPUT = open(embedding_filename, "a")

        else:
            OUTPUT = open(embedding_filename, "w")
            mu_header_list = ["mu_"+str(i+1) for i in range(model.n_latent)]
            log_sigma_header_list = ["log_sigma_"+str(i+1) for i in range(model.n_latent)]

            header_list = mu_header_list + log_sigma_header_list
            OUTPUT.write("update_num,name,"+",".join(header_list)+"\n")


        batch_order = np.arange(len(self.x_train_name_list))

        for i in range(0,len(self.x_train_name_list),minibatch_size):
            batch_index = batch_order[i:i+minibatch_size]
            one_hot_seqs = self.x_train[batch_index]
            batch_mu, batch_log_sigma  = model.recognize(one_hot_seqs)

            for j,idx in enumerate(batch_index.tolist()):
                sample_mu = [str(val) for val in batch_mu[j].tolist()]
                sample_log_sigma = [str(val) for val in batch_log_sigma[j].tolist()]
                sample_name = self.x_train_name_list[idx]
                out_line = [str(update_num),sample_name]+sample_mu+sample_log_sigma
                if verbose:
                    print ("\t".join(out_line))
                OUTPUT.write(",".join(out_line)+"\n")

        OUTPUT.close()

    def get_elbo_samples(self, model, N_pred_iterations=100, minibatch_size=2000):

        self.prediction_matrix = np.zeros((self.one_hot_mut_array_with_wt.shape[0],N_pred_iterations))

        batch_order = np.arange(self.one_hot_mut_array_with_wt.shape[0])

        for i in range(N_pred_iterations):
            np.random.shuffle(batch_order)

            for j in range(0,self.one_hot_mut_array_with_wt.shape[0],minibatch_size):

                batch_index = batch_order[j:j+minibatch_size]
                batch_preds, _, _ = model.all_likelihood_components(self.one_hot_mut_array_with_wt[batch_index])

                for k,idx_batch in enumerate(batch_index.tolist()):
                    self.prediction_matrix[idx_batch][i]= batch_preds[k]

def gen_job_string(data_params, model_params):
    """
        Generates a unique job string given data and model parameters.
        This is used later as an identifier for the
                saved model weights and figures
        Parameters
        ------------
        data_params: dictionary of parameters for the data class
        model_params: dictionary of parameters for the model class

        Returns
        ------------
        job string denoting parameters of run
    """

    written_out_vals = ["n_latent"]
    layer_num_list = ["zero","one","two","three","four"]

    encoder_architecture = []
    decoder_architecture = []

    for layer_num in layer_num_list:
        if "encode_dim_"+layer_num in model_params:
            encoder_architecture.append(model_params["encode_dim_"+layer_num])
        if "decode_dim_"+layer_num in model_params:
            decoder_architecture.append(model_params["decode_dim_"+layer_num])

        written_out_vals += ["encode_dim_"+layer_num, "decode_dim_"+layer_num]

    n_latent = model_params["n_latent"]

    encoder_architecture_str = "-".join([str(size) for size in encoder_architecture])
    decoder_architecture_str = "-".join([str(size) for size in decoder_architecture])

    job_str = "vae_output_encoder-"+encoder_architecture_str+"_Nlatent-"+str(n_latent)\
        +"_decoder-"+decoder_architecture_str

    job_id_list = []
    for data_id,data_val in sorted(data_params.items()):
        if data_id not in written_out_vals:
            if str(type(data_val)) == "<type 'list'>":
                job_id_list.append(data_id+"-"+"-".join([str(val) for val in data_val]))
            else:
                job_id_list.append(data_id+"-"+str(data_val))


    for model_id,model_val in sorted(model_params.items()):
        if model_id not in written_out_vals:
            if str(type(model_val)) == "<type 'list'>":
                job_id_list.append(model_id+"-"+"-".join([str(val) for val in model_val]))
            else:
                job_id_list.append(model_id+"-"+str(model_val))


    return job_str+"_"+"_".join(job_id_list)
