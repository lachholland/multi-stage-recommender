import torch
import kaggle
import os

def data_init():
    if os.path.exists('./data/[file-name].csv'):
        return True
    else:
        kaggle.api.dataset_download_files("h-and-m-personalized-fashion-recommendations", path="./data", unzip=True)

def split_data(data, train_decimal=0.6, val_decimal=0.2):
    train_size = int(train_decimal * len(data))
    val_size = int(val_decimal * len(data))
    test_size = len(data) - train_size - val_size 
    train_data, val_data, test_data = torch.utils.data.random_split(data, [train_size, val_size, test_size])
    return train_data, val_data, test_data