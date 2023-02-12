import torch
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
import os
from ..datasets.CustomDataset import CustomDataset
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


def split_data(train_decimal=0.6, val_decimal=0.2): 
    data = pd.read_csv(os.path.join(os.getcwd(), 'data', 'data_files', 'transactions.csv')) 
    train_size = int(train_decimal * len(data))
    val_size = int(val_decimal * len(data))
    test_size = len(data) - train_size - val_size 
    train_data, val_data, test_data = torch.utils.data.random_split(data, [train_size, val_size, test_size])
    return train_data, val_data, test_data


def CustomDatasetCreator(transactions_train_data):
    transform=lambda x:customer_lookup(transactions_train_data)[0].__getitem__(x)
    target_transform=lambda y:article_lookup(transactions_train_data)[0].__getitem__(str(y))
    dataset = CustomDataset(transactions_train_data,transform=transform,target_transform=target_transform)
    article_vocab_size=article_lookup(transactions_train_data)[1]
    customer_vocab_size=customer_lookup(transactions_train_data)[1]
    return [dataset,article_vocab_size,customer_vocab_size]


def article_lookup(train_df:pd.DataFrame):
    unique_article_ids=train_df.article_id.unique() # list of unique article_ids found in training dataset
    vocab=build_vocab_from_iterator([yield_tokens(unique_article_ids)], specials=["<unk>"]) # vocab is a torchtext.vocab.Vocab object
    article_vocab_size=len(unique_article_ids)+1
    return [vocab,article_vocab_size]


def customer_lookup(train_df:pd.DataFrame):
    unique_customer_ids=train_df.customer_id.unique() # list of unique customer_ids found in training dataset
    vocab=build_vocab_from_iterator([yield_tokens(unique_customer_ids)], specials=["<unk>"]) # vocab is a torchtext.vocab.Vocab object
    customer_vocab_size=len(unique_customer_ids)+1
    return [vocab,customer_vocab_size]


def yield_tokens(unique_ids):
    for id in unique_ids:
        yield str(id)


def DataLoaderCreator(dataset,batch_size,shuffle_dataset=True,random_seed=42):
    train_split=0.6
    val_split = 0.2
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_end = int(train_split * dataset_size)
    val_end = int(val_split * dataset_size)
    train_indices = indices[:train_end]
    val_indices=indices[train_end:val_end]
    test_indices=indices[val_end:]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
    val_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)
    test_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=test_sampler)
    return train_data_loader,val_data_loader, test_data_loader