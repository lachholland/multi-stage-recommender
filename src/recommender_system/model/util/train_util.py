import torch
import torch.nn as nn

def train_step(recommender_system, data, criterion, epoch, train_loss_history, learning_rate):
    inputs, outputs = data 
    logits = recommender_system(inputs, outputs)
    mapping = mapping_labels(outputs)
    loss = criterion(logits, mapping)
    optimizer = torch.optim.Adam(recommender_system.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    accuracy = (logits == (mapping)).sum().item()
    result = {'train_loss': loss, 'train_accuracy': accuracy, 'epoch': epoch}
    train_loss_history.append(result)
    return result


def val_step(recommender_system, data, criterion, epoch, val_loss_history):
    inputs, outputs = data
    sorted, indices=torch.sort(outputs)
    mapping = mapping_labels(outputs)
    logits = recommender_system(inputs,sorted)
    loss = criterion(logits, mapping )
    _, predicted = torch.max(logits, 1)
    accuracy = (predicted == (mapping)).sum()
    result = {'val_loss': loss, 'val_accuracy': accuracy, 'epoch': epoch} 
    val_loss_history.append(result)
    return result


def test_step(recommender_system, data, criterion, epoch, test_loss_history):
    inputs, outputs = data
    sorted, indices=torch.sort(outputs)
    mapping = mapping_labels(outputs)
    logits = recommender_system(inputs,sorted)
    loss = criterion(logits, mapping )
    _, predicted = torch.max(logits, 1)
    accuracy = (predicted == (mapping)).sum()
    result = {'test_loss': loss, 'test_accuracy': accuracy, 'epoch': epoch}
    test_loss_history.append(result) 
    return result


def train_recommender_system(recommender_system, train_dataloader, val_dataloader, test_dataloader, epochs=10):
    learning_rate = 0.0001
    train_loss_history = []
    val_loss_history = []
    test_loss_history = []
    criterion=nn.CrossEntropyLoss()
    best_val_accuracy=0.0
    for epoch in range(epochs):
        running_train_loss=0.0
        running_val_loss=0.0
        running_test_loss=0.0
        running_train_accuracy=0.0
        running_val_accuracy=0.0
        running_test_accuracy=0.0
        total_predictions_val=0
        total_predictions_test=0
       
        # TRAINING
        for i,data in train_dataloader:
            result = train_step(recommender_system, data, criterion, epoch, train_loss_history, learning_rate)
            running_train_loss += result['train_loss'].item()
            running_train_accuracy += result['train_accuracy'].item()
        train_loss_value = running_train_loss/len(train_dataloader) 
        print('train loss: ', train_loss_value)
        test_accuracy_value = (100 * running_train_accuracy / total_predictions_test)
        print('test accuracy: ', test_accuracy_value)

        # VALIDATION
        with torch.no_grad(): 
            recommender_system.eval() 
            for data in val_dataloader:
                result = val_step(recommender_system, data, criterion, epoch, val_loss_history)
                running_val_loss += result['val_loss'].item()
                running_val_accuracy += result['val_accuracy'].item()
                total_predictions_val += data[1].size(0)
        val_loss_value = running_val_loss/len(val_dataloader) 
        print('val loss: ', val_loss_value)
        val_accuracy_value = (100 * running_val_accuracy / total_predictions_val)
        print('val accuracy: ', val_accuracy_value)

        if val_accuracy_value > best_val_accuracy: 
            torch.save(recommender_system.state_dict(), 'recommender_val.pt') 
        best_val_accuracy = val_accuracy_value 
        
        # TESTING
        with torch.no_grad():
            recommender_system.eval()
            for data in test_dataloader:
                result = test_step(recommender_system, data, criterion, epoch, test_loss_history)
                running_test_loss += result['test_loss'].item()
                running_test_accuracy += result['test_accuracy'].item()
                total_predictions_test += data[1].size(0)
        test_loss_value = running_test_loss/len(test_dataloader) 
        print('test loss: ', test_loss_value)
        test_accuracy_value = (100 * running_test_accuracy / total_predictions_test)
        print('test accuracy: ', test_accuracy_value)


# maps labels in the batch to integers in [0, batch_size])
def mapping_labels(item_train):
    unique_labels=torch.unique(item_train)
    mapped_labels=torch.zeros(item_train.size(dim=0))
    for i in range(mapped_labels.size(dim=0)):
        mapped_labels[i]=(unique_labels==item_train[i]).nonzero(as_tuple=True)[0]
    return mapped_labels.long()


