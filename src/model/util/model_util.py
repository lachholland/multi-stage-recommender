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