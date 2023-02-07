from recommender_system.data.util import data_util
from recommender_system.model.util import train_util
from recommender_system.model import RecommenderSystem
from recommender_system.model import RecommenderTowerModel
from torch.utils.data import DataLoader
from recommender_system.data.datasets import CustomDataset
import pandas as pd

def main():
    transactions_train_data=pd.read_csv(r'.\recommender_system\data\transactions_train.csv')
    item_model = RecommenderTowerModel.RecommenderTowerModel()
    user_model = RecommenderTowerModel.RecommenderTowerModel()
    recommender_system = RecommenderSystem.RecommenderSystem(user_model, item_model)
    complete_customdataset = data_util.CustomDatasetCreator(transactions_train_data)
    train_data_loader,validation_loader=data_util.DataLoaderCreator(complete_customdataset,batch_size=64,splits=0.2)
    train_util.train_recommender_system(recommender_system, train_data_loader, epochs=10)

if __name__ == '__main__':
    main()
