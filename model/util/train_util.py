import torch
import torch.nn as nn
from data import util as data_util
from model import RecommenderSystem
from model import RecommenderTowerModel

def train_step(recommender_system, user_train, item_train, epoch):
    logits = recommender_system(user_train, item_train)
    criterion = nn.CrossEntropyLoss()
    loss_value = criterion(logits, item_train)
    optimizer = torch.optim.Adam(recommender_system.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
    return {'loss': loss_value, 'epoch': epoch}

def train_recommender_system(recommender_system, dataloader, epochs=10):
    loss_history = []
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            result = train_step(recommender_system, inputs, labels, epoch)
            loss_history.append(result)
            result = train_step(recommender_system, inputs, labels)
            loss_history.append(result)

def train_start():
    train_data_loader = data_util.dataset_init()
    item_model = RecommenderTowerModel()
    user_model = RecommenderTowerModel()
    recommender_system = RecommenderSystem(user_model, item_model)
    train_recommender_system(recommender_system, train_data_loader, epochs=10)
