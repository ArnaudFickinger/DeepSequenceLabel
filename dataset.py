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