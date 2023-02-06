import torch.nn as nn

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
