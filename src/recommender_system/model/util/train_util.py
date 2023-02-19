import torch
import torch.nn as nn
from .model_util import mapping_labels

def train_step(recommender_system, data, criterion, epoch, learning_rate):
    inputs, outputs = data 
    logits = recommender_system(inputs, outputs)
    mapping = mapping_labels(outputs)
    loss = criterion(logits, mapping)
    optimizer = torch.optim.Adam(recommender_system.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    result = {'train_loss': loss, 'epoch': epoch}
    return result

def train_recommender_system(recommender_system, train_dataloader, epochs=10):
    learning_rate = 0.0001
    criterion=nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_train_loss=0.0
        total_predictions_train=0
        for data in train_dataloader:
            result = train_step(recommender_system, data, criterion, epoch, learning_rate)
            running_train_loss += result['train_loss'].item()
            total_predictions_train += data[1].size(0)
        train_loss_value = running_train_loss/len(train_dataloader) 
        print(f'epoch: {epoch}, train loss: {train_loss_value}')
 