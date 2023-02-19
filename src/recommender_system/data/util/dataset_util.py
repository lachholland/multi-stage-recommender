import torch
from ..datasets.CustomDataset import CustomDataset
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from .transform_util import user_lookup, item_lookup
import pandas as pd

def CustomDatasetCreator(transactions_train_data):
    transform=lambda x:user_lookup(transactions_train_data)[0].__getitem__(x)
    target_transform=lambda y:item_lookup(transactions_train_data)[0].__getitem__(str(y))
    dataset = CustomDataset(transactions_train_data,transform=transform,target_transform=target_transform)
    article_vocab_size=user_lookup(transactions_train_data)[1]
    customer_vocab_size=item_lookup(transactions_train_data)[1]
    return [dataset,article_vocab_size,customer_vocab_size]

def dataLoaderCreator(dataset,batch_size,shuffle_dataset=True,random_seed=42):
    train_split=0.6
    val_split = 0.2
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_end = int(np.floor(train_split * dataset_size))
    val_end = int(np.floor((train_split + val_split) * dataset_size))
    train_indices = indices[:train_end]
    val_indices=indices[train_end:val_end]
    test_indices=indices[val_end:]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    test_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    return train_data_loader,val_data_loader, test_data_loader
