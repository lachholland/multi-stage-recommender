import torch
import torch.nn as nn
from .model_util import mapping_labels

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

def validate_recommender_system(recommender_system, val_dataloader, epochs=10):
    val_loss_history = []
    criterion=nn.CrossEntropyLoss()
    best_val_accuracy=0.0
    for epoch in range(epochs):
        running_val_loss=0.0
        running_val_accuracy=0.0
        total_predictions_val=0
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
