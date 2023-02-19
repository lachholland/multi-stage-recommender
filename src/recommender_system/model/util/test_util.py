import torch
import torch.nn as nn
from .model_util import mapping_labels

def test_step(recommender_system, data, criterion, epoch):
    inputs, outputs = data
    sorted, indices=torch.sort(outputs)
    mapping = mapping_labels(outputs)
    logits = recommender_system(inputs,sorted)
    loss = criterion(logits, mapping )
    _, predicted = torch.max(logits, 1)
    accuracy = (predicted == (mapping)).sum()
    result = {'test_loss': loss, 'test_accuracy': accuracy, 'epoch': epoch}
    return result

def test_recommender_system(recommender_system, test_dataloader, epochs=10):
    criterion=nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_test_loss=0.0
        running_test_accuracy=0.0
        total_predictions_test=0
        with torch.no_grad():
            recommender_system.eval()
            for data in test_dataloader:
                result = test_step(recommender_system, data, criterion, epoch)
                running_test_loss += result['test_loss'].item()
                running_test_accuracy += result['test_accuracy'].item()
                total_predictions_test += data[1].size(0)
        test_loss = running_test_loss/len(test_dataloader) 
        test_accuracy = (100 * running_test_accuracy / total_predictions_test)
        print(f'epoch: {epoch}, test loss: {test_loss}, test accuracy: {test_accuracy}')