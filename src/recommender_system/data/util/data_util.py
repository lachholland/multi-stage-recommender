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
    transactions_train_dat = pd.read_csv(r'C:\Users\navpa\recommender_system\multi_stage_recommender.git\data\transactions_train.csv')
    #pd.read_csv('data/transactions_train.csv')
    transform=lambda x:customer_lookup(transactions_train_dat).__getitem__(x)
    target_transform=lambda y:article_lookup(transactions_train_dat).__getitem__(y)
    dataset = CustomDataset(transactions_train_dat,transform=transform,target_transform=target_transform)
    return dataset



def article_lookup(train_df:pd.DataFrame):
    unique_article_ids=train_df.article_id.unique() # list of unique article_ids found in training dataset
    vocab=build_vocab_from_iterator(yield_tokens(unique_article_ids), specials=["<unk>"]) # vocab is a torchtext.vocab.Vocab object
    article_vocab_size=len(unique_article_ids)+1
    print(f'article vocab size = {article_vocab_size}')
    return vocab

def customer_lookup(train_df:pd.DataFrame):
    unique_customer_ids=train_df.customer_id.unique() # list of unique customer_ids found in training dataset
    vocab=build_vocab_from_iterator(yield_tokens(unique_customer_ids), specials=["<unk>"]) # vocab is a torchtext.vocab.Vocab object
    customer_vocab_size=len(unique_customer_ids)+1
    print(f'article vocab size = {customer_vocab_size}')
    return vocab

def yield_tokens(unique_ids):
    for id in unique_ids:
        yield id

def DataLoaderCreator(dataset,batch_size,splits,shuffle_dataset=True,random_seed=42):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(splits * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices = indices[split:]
    val_indices=indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)
    return train_loader,validation_loader

#train_loader,validation_loader=DataLoaderCreator(CustomDatasetCreator(pd.read_csv(r"C:\Users\navpa\recommender_system\multi_stage_recommender.git\data\transactions_train.csv")),64,0.2)
#print(len(train_loader))