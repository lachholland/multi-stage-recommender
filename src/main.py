import pandas as pd
from recommender_system.data.util import data_util
from recommender_system.model.util import train_util
from recommender_system.model import RecommenderSystem, RecommenderTowerModel
from recommender_system.model.util import val_util
from recommender_system.model.util import test_util

def main():
    transactions_train_data=pd.read_csv(r'./transactions_train.csv')
    testing_df=transactions_train_data.head(5000)
    creator_output=data_util.CustomDatasetCreator(testing_df)
    complete_customdataset = creator_output[0]
    item_vocab_size=creator_output[1]
    user_vocab_size=creator_output[2]
    item_model = RecommenderTowerModel.RecommenderTowerModel(vocab_size=item_vocab_size,embedding_dimension=256)
    user_model = RecommenderTowerModel.RecommenderTowerModel(vocab_size=user_vocab_size,embedding_dimension=256)
    recommender_system = RecommenderSystem.RecommenderSystem(user_model, item_model)
    train_data_loader,val_data_loader,test_data_loader=data_util.DataLoaderCreator(complete_customdataset,batch_size=64)
    train_util.train_recommender_system(recommender_system,train_data_loader,val_data_loader, test_data_loader, epochs=10)
    val_util.validate_recommender_system(recommender_system, val_data_loader, epochs=10)
    test_util.test_recommender_system(recommender_system, test_data_loader, epochs=10)

if __name__ == '__main__':
    main()
    