import torch
import kaggle
import os
from torchtext.vocab import build_vocab_from_iterator
import torchtext
import pandas as pd
from datasets import CustomDataset

def split_data(data, train_decimal=0.6, val_decimal=0.2):
    train_size = int(train_decimal * len(data))
    val_size = int(val_decimal * len(data))
    test_size = len(data) - train_size - val_size 
    train_data, val_data, test_data = torch.utils.data.random_split(data, [train_size, val_size, test_size])
    return train_data, val_data, test_data

def article_lookup(train_df:pd.DataFrame):
    unique_article_ids=train_df.article_id.unique() # list of unique article_ids found in training dataset
    vocab=build_vocab_from_iterator(yield_tokens(unique_article_ids), specials=["<unk>"]) # vocab is a torchtext.vocab.Vocab object
    article_vocab_size=len(unique_article_ids)+1
    print(f'article vocab size = {article_vocab_size}')
    return vocab

def customer_lookup(train_df:pd.DataFrame):
    unique_customer_ids=train_df.article_id.unique() # list of unique customer_ids found in training dataset
    vocab=build_vocab_from_iterator(yield_tokens(unique_customer_ids), specials=["<unk>"]) # vocab is a torchtext.vocab.Vocab object
    customer_vocab_size=len(unique_customer_ids)+1
    print(f'article vocab size = {customer_vocab_size}')
    return vocab

def yield_tokens(unique_ids):
    for id in unique_ids:
        yield id

def dataset_init():
    articles_data = pd.read_csv('data/articles.csv')
    customers_data = pd.read_csv('data/customers.csv')
    transactions_train_data = pd.read_csv('data/transactions_train.csv')
    pd.read_csv('data/transactions_train.csv')
    dataset = CustomDataset(articles_data, customers_data, transactions_train_data)
    dataloader = torch.utils.data.Dataloader(dataset, batch_size=2, shuffle=True)
    return dataloader

