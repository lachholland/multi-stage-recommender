import torch
import torch.nn as nn

def train_step(recommender_system, user_train, item_train, epoch):
    logits = recommender_system(user_train, item_train)
    criterion = nn.CrossEntropyLoss()
    loss_value = criterion(logits, item_train)
    optimizer = torch.optim.Adam(recommender_system.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
    return {'loss': loss_value, 'epoch': epoch}

def train_recommender_system(recommender_system, train_dataloader,val_dataloader, epochs=10):
    loss_history = []
    criterion=nn.CrossEntropyLoss()
    best_accuracy=0.0
    for epoch in range(epochs):
        running_val_loss=0.0
        running_train_loss=0.0
        running_accuracy=0.0
        total=0
        for i,data in enumerate(train_dataloader,0):
            inputs,labels=data
            print(inputs[:10])
            result = train_step(recommender_system, inputs, labels, epoch)
            loss_history.append(result)
            running_loss += running_loss['loss'].item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        #result = train_step(recommender_system, inputs, labels)
        #loss_history.append(result