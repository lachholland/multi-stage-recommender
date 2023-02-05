import torch
from data import util as data_util
from model import util as model_util
from model import RecommenderSystem
from model import RecommenderTowerModel

def main():
    dataloader = data_util.dataset_init()
    item_model = RecommenderTowerModel()
    user_model = RecommenderTowerModel()
    recommender_system = RecommenderSystem(user_model, item_model)
    model_util.train_recommender_system(recommender_system, dataloader, epochs=10)

if __name__ == '__main__':
    main()
