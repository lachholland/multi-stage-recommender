import torch
import torch.nn as nn

def train_step(recommender_system, user_train, item_train, epoch):
    logits = recommender_system(user_train, item_train)
    #print(item_train)
    #print(logits.size())
    mapped_labels=mapping_labels(item_train)
    criterion = nn.CrossEntropyLoss()
    loss_value = criterion(logits, mapped_labels)
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
        running_loss=0.0
        for i,data in enumerate(train_dataloader,0):
            #print(i)
            inputs,labels=data
            #print(labels)
            result = train_step(recommender_system, inputs, labels, epoch)
            loss_history.append(result)
            running_loss += result['loss'].item()
            if i % 20 == 19:    
                
                # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        print(loss_history)
        loss_history=[]
        #result = train_step(recommender_system, inputs, labels)
        #loss_history.append(result


"""
        this function maps our labels in the batch to integers in [0,64)]

"""
def mapping_labels(item_train):
    unique_labels=torch.unique(item_train)
    mapped_labels=torch.zeros(item_train.size(dim=0))
    for i in range(mapped_labels.size(dim=0)):
        mapped_labels[i]=(unique_labels==item_train[i]).nonzero(as_tuple=True)[0]
    #print(mapped_labels.long())
    return mapped_labels.long()


