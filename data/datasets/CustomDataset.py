import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, user_data, item_data, transactions_train_data, transform=None):
        self.user_data = user_data 
        self.item_data = item_data
        self.transactions_train_data = transactions_train_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        return x
 