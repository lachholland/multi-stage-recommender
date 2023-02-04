import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.item_data = pd.read_csv('./data/[file-name].csv') 
        self.user_data = pd.read_csv('./data/[file-name].csv')
        self.targets = torch.LongTensor(targets)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        return x

    
   
   
   
   
   

   
   
   
   
   

   
   