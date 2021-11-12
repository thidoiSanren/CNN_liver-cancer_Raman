from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np


def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = (data - minVals) / ranges
    return normData

def z_score(data):
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)

    return data





class TrainSets(Dataset):
    def __init__(self):
        train_data = np.loadtxt('./data/liver_all_train1.txt', delimiter='\t')
        self.x = train_data[:, 1:]
        self.y = train_data[:, 0]
        self.len = len(train_data)
        self.x = z_score(self.x)
        # self.x = noramlization(self.x)
    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.len


class ValidateSets(Dataset):
    def __init__(self):
        validata_data = np.loadtxt('./data/liver_all_val_cuttest.txt', delimiter='\t')
        self.x = validata_data[:, 1:]
        self.y = validata_data[:, 0]
        self.len = len(validata_data)
        self.x = z_score(self.x)
        # self.x = noramlization(self.x)
    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.len

class TestdataSets(Dataset):
    def __init__(self):
        test_data = np.loadtxt('./data/liver_all_test1.txt', delimiter='\t')
        self.x = test_data[:, 1:]
        self.y = test_data[:, 0]
        self.len = len(test_data)
        self.x = z_score((self.x))
    def __getitem__(self, item):
        return  self.x[item], self.y[item]

    def __len__(self):
        return self.len
