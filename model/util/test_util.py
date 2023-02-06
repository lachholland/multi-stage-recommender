import torch
import torch.nn as nn
from data import util as data_util
from model import RecommenderSystem

def test_step(recommender_system, user_train, item_train, epoch):
    logits = recommender_system(user_train, item_train)
    criterion = nn.CrossEntropyLoss()
    loss_value = criterion(logits, item_train)
    return {'loss': loss_value, 'epoch': epoch}

def test_recommender_system(recommender_system, dataloader):
    loss_history = []
    for inputs, labels in dataloader:
            result = test_step(recommender_system, inputs, labels)
            loss_history.append(result)

def test_start(model: RecommenderSystem):
    test_data_loader = data_util.dataset_init()
    test_recommender_system(model, test_data_loader) 