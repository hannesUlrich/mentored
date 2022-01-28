import torch
import numpy as np
from torch.utils.data import Dataset


class data(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx].astype(np.int32)), self.y[idx], np.count_nonzero(self.X[idx])
