from recommender_system.data.util import data_util
from recommender_system.model.util import train_util
from recommender_system.model import RecommenderSystem
from recommender_system.model import RecommenderTowerModel
from torch.utils.data import DataLoader
from recommender_system.data.datasets import CustomDataset
import pandas as pd

def main():
    print(1)
    transactions_train_data=pd.read_csv(r'.\recommender_system\data\transactions_train.csv')
    #testing_df=transactions_train_data.head(10)
    complete_customdataset = data_util.CustomDatasetCreator(transactions_train_data)[0]
    item_vocab_size=data_util.CustomDatasetCreator(transactions_train_data)[1]
    print(item_vocab_size)
    user_vocab_size=data_util.CustomDatasetCreator(transactions_train_data)[2]
    print(user_vocab_size)
    item_model = RecommenderTowerModel.RecommenderTowerModel(vocab_size=item_vocab_size,embedding_dimension=10)
    user_model = RecommenderTowerModel.RecommenderTowerModel(vocab_size=user_vocab_size,embedding_dimension=10)
    recommender_system = RecommenderSystem.RecommenderSystem(user_model, item_model)
    train_data_loader,validation_loader=data_util.DataLoaderCreator(complete_customdataset,batch_size=64,splits=0.2)
    train_util.train_recommender_system(recommender_system,train_data_loader,validation_loader,epochs=10)

    """
    testing_df=transactions_train_data.head(50)
    complete_customdataset = data_util.CustomDatasetCreator(testing_df)[0]
    item_vocab_size=data_util.CustomDatasetCreator(testing_df)[1]
    user_vocab_size=data_util.CustomDatasetCreator(testing_df)[2]
    item_model = RecommenderTowerModel.RecommenderTowerModel(vocab_size=item_vocab_size,embedding_dimension=10)
    user_model = RecommenderTowerModel.RecommenderTowerModel(vocab_size=user_vocab_size,embedding_dimension=10)
    recommender_system = RecommenderSystem.RecommenderSystem(user_model, item_model)
    train_data_loader,validation_loader=data_util.DataLoaderCreator(complete_customdataset,batch_size=64,splits=0.2)
    train_util.train_recommender_system(recommender_system, train_data_loader, epochs=10)

    """

if __name__ == '__main__':
    main()
