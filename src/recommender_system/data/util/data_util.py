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
    print(f'article vocab size = {article_vocab_size}')
    return [vocab,article_vocab_size]

def customer_lookup(train_df:pd.DataFrame):
    unique_customer_ids=train_df.customer_id.unique() # list of unique customer_ids found in training dataset
    #print(unique_customer_ids)
    vocab=build_vocab_from_iterator([yield_tokens(unique_customer_ids)], specials=["<unk>"]) # vocab is a torchtext.vocab.Vocab object
    customer_vocab_size=len(unique_customer_ids)+1
    print(f'customer vocab size = {customer_vocab_size}')
    return [vocab,customer_vocab_size]

def yield_tokens(unique_ids):
    for id in unique_ids:
        #print(type(str(id)))
        yield str(id)

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

#testing here

train_df=pd.read_csv(r"C:\Users\navpa\recommender_system\multi_stage_recommender.git\src\recommender_system\data\transactions_train.csv")
testing_df=train_df.head(5)
print(testing_df)
print(len(CustomDatasetCreator(testing_df)[0]))
train_loader,validation_loader=DataLoaderCreator(CustomDatasetCreator(testing_df)[0],64,0.2)
train_features, train_labels = next(iter(train_loader))
for i in range(len(train_features)):
    print(train_features[i],train_labels[i])
print(article_lookup(testing_df)[0].get_stoi())
print(customer_lookup(testing_df)[0].get_stoi())