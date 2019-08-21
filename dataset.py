###
'''
April 2019
Code by: Arnaud Fickinger
'''
###

import torch
from utils import *
import numpy as np


class Dataset_labelled(torch.utils.data.Dataset):
    def __init__(self, labeled_mut, labels):
        super(Dataset_labelled, self).__init__()
        self.mutations = np.array(labeled_mut)
        self.labels = np.array(labels)

    def __getitem__(self, index):
        return self.mutations[index], self.labels[index]

    def __len__(self):
        return len(self.mutations)

class Dataset_unlabelled(torch.utils.data.Dataset):
    def __init__(self, unlabeled_mut):
        super(Dataset_unlabelled, self).__init__()
        self.data = np.array(unlabeled_mut)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_helper):
        super(Dataset, self).__init__()

        all_data = dataset_helper.x_train
        self.data = all_data
        # total_len = all_data.shape[0]
        # if isTrain:
        #     self.len = int(0.8 * total_len)
        #     self.data = all_data[:self.len,:,:]
        # else:
        #     self.len = total_len - int(0.8 * total_len)
        #     self.data = all_data[int(0.8 * total_len):,:,:]


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class Dataset_merge(torch.utils.data.Dataset):
    def __init__(self, dataset_helper, seq_hg):
        super(Dataset_merge, self).__init__()

        # print(seq_hg.shape)
        # print(dataset_helper.x_train.shape)

        all_data = np.concatenate((seq_hg, dataset_helper.x_train), axis=0)
        # print(all_data.shape)
        # print(len(all_data))
        self.data = all_data
        # total_len = all_data.shape[0]
        # if isTrain:
        #     self.len = int(0.8 * total_len)
        #     self.data = all_data[:self.len,:,:]
        # else:
        #     self.len = total_len - int(0.8 * total_len)
        #     self.data = all_data[int(0.8 * total_len):,:,:]


    def __getitem__(self, index):
        # print(index)
        return self.data[index]

    def __len__(self):
        return len(self.data)