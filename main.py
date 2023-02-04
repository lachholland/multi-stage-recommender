from data.util import util
from model import util as model_util
from model import RecommenderSystem
from model import RecommenderTowerModel

def main():
    util.data_init()
    user_model = RecommenderTowerModel()
    item_model = RecommenderTowerModel()
    tower_model = RecommenderSystem(user_model, item_model)
    model.util.train_model(tower_model, epochs=10)

if __name__ == '__main__':
    main()
