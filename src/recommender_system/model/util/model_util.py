import torch

# maps labels in the batch to integers in [0, batch_size])
def mapping_labels(item_train):
    unique_labels=torch.unique(item_train)
    mapped_labels=torch.zeros(item_train.size(dim=0))
    for i in range(mapped_labels.size(dim=0)):
        mapped_labels[i]=(unique_labels==item_train[i]).nonzero(as_tuple=True)[0]
    return mapped_labels.long()
