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
        running_accuracy=0.0
        total=0
        inputs, labels=next(iter(train_dataloader))
        result = train_step(recommender_system, inputs, labels, epoch)
        loss_history.append(result)
        #result = train_step(recommender_system, inputs, labels)
        #loss_history.append(result)
        with torch.no_grad(): 
            recommender_system.eval() 
            for data in val_dataloader: 
               inputs, outputs = data 
               predicted_outputs = recommender_system(inputs) 
               val_loss = criterion(predicted_outputs, outputs) 
             
             # The label with the highest value will be our prediction 
               _, predicted = torch.max(predicted_outputs, 1) 
               running_val_loss += val_loss.item()  
               total += outputs.size(0) 
               running_accuracy += (predicted == outputs).sum().item()
        # Calculate validation loss value 
        val_loss_value = running_val_loss/len(val_dataloader) 
                
        # Calculate accuracy as the number of correct predictions in the validation batch divided by the total number of predictions done.  
        accuracy = (100 * running_accuracy / total)
        print(accuracy)

        # Save the model if the accuracy is the best 
        #if accuracy > best_accuracy: 
         #   saveModel() 
          #  best_accuracy = accuracy 
    print(loss_history)