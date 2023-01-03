import torch
from typing import Tuple
from torch import Tensor 
from torch.utils.data import TensorDataset
import numpy as np


def mnist() -> Tuple[Tensor, Tensor]:
    # exchange with the corrupted mnist dataset
    path = "data/corruptmnist/"
    with np.load(path +'test.npz') as test_data:
        test_images = torch.from_numpy(test_data['images'])
        test_labels = torch.from_numpy(test_data['labels'])
        test = TensorDataset(test_images, test_labels)

    file_list = [path+'train_0.npz', path+'train_1.npz', path+'train_2.npz', path+'train_3.npz', path+'train_4.npz']
    data_all = [np.load(fname) for fname in file_list]
    merged_train_data = {}
    for data in data_all:
        [merged_train_data.update({k: v}) for k, v in data.items()]

    train_images = torch.from_numpy(merged_train_data['images'])
    train_labels = torch.from_numpy(merged_train_data['labels'])
    train = TensorDataset(train_images, train_labels)
    return train, test
