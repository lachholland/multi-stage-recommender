import torch
import torch.nn as nn
from model.Recommender import Recommender
from model.Tower import Tower
import datasets.ItemDataset as ItemDataset 
import datasets.UserDataset as UserDataset

def custom_cross_entropy_loss(logits, true_labels, training):
    batch_size, nb_candidates = logits.shape
    if training:
        label_probs = 
        logits -= torch.log(label_probs)
        true_labels = tf.range(0, batch_size)
    loss = nn.CrossEntropyLoss(logits, true_labels)
    return {'loss': loss, 'logits': logits} 

def train_step(model, user_train, item_train, epoch):
    logits = model(user_train, item_train)
    criterion = nn.CrossEntropyLoss() 
    loss_value = criterion(logits, item_train)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
    return {'loss': loss_value, 'epoch': epoch}

def train_model(model):
    item_train = ItemDataset(csv_file='data/item_train.csv')
    user_train = UserDataset(csv_file='data/user_train.csv')
    loss_history = []
    for epoch in range(10):
        result = train_step(model,user_train, item_train, epoch)
        loss_history.append(result)

def main():
    user_model = Tower()
    item_model = Tower()
    tower_model = Recommender(user_model, item_model)
    train_model(tower_model)

if __name__ == '__main__':
   main()