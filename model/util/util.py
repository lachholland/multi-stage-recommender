import torch
import torch.nn as nn

def custom_cross_entropy_loss(logits, true_labels, training):
    batch_size, nb_candidates = logits.shape
    if training:
        label_probs = '' 
        logits -= torch.log(label_probs)
        true_labels = torch.range(0, batch_size) #?
    loss = nn.CrossEntropyLoss(logits, true_labels)
    return loss

def train_step(recommender_system, user_train, item_train, epoch):
    logits = recommender_system(user_train, item_train)
    criterion = nn.CrossEntropyLoss()
    loss_value = criterion(logits, item_train)
    optimizer = torch.optim.Adam(recommender_system.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
    return {'loss': loss_value, 'epoch': epoch}

def train_recommender_system(recommender_system, user_train, item_train, epochs=10):
    loss_history = []
    for epoch in range(epochs):
        result = train_step(recommender_system, user_train, item_train, epoch)
        loss_history.append(result)
    