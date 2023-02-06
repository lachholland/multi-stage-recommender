from recommender_system.data.util import data_util
from recommender_system.model.util import train_util
from recommender_system.model import RecommenderSystem
from recommender_system.model import RecommenderTowerModel
from torch.utils.data import DataLoader
from recommender_system.data.datasets import CustomDataset

def main():
    item_model = RecommenderTowerModel()
    user_model = RecommenderTowerModel()
    recommender_system = RecommenderSystem(user_model, item_model)
    train_data, test_data, val_data = data_util.split_data()
    train_data_loader = data_util.dataset_init(train_data) 
    train_util.train_recommender_system(recommender_system, train_data_loader, epochs=10)

if __name__ == '__main__':
    main()
