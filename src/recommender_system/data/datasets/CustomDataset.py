from torch.utils.data import Dataset
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, transactions__data:pd.DataFrame, transform=None,target_transform=None):
        self.transactions_data = transactions__data
        self.transform=transform  # x value transform
        self.target_transform=target_transform  #y value transform

    def __len__(self):
        return len(self.transactions_data)

    def __getitem__(self, index):
        user=self.transactions_data['customer_id'][index] #fetches the x value (user)
        label=self.transactions_data['article_id'][index] #fetches they y value (article)
        if self.transform:
            user=self.transform(user)  #transforms the x value if transformation exists
        if self.target_transform:
            label=self.target_transform(label)  #transforms the y value if target_transformation exists
        return user,label
 