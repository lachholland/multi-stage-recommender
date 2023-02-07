import torch
import torch.nn as nn
from .RecommenderTowerModel import RecommenderTowerModel

class RecommenderSystem(nn.Module):
    def __init__(self, user_model: RecommenderTowerModel, item_model: RecommenderTowerModel):
        super(RecommenderSystem).__init__()
        self.user_model = user_model
        self.item_model = item_model
       
    def forward(self, user_inputs, item_inputs):
        user_embeddings = self.user_model(user_inputs['user_id'])
        if self.training:
            item_embeddings = self.item_model(item_inputs['item_id'])
        else:
            item_embeddings = self.item_model(self.all_items)
        return torch.matmul(user_embeddings, torch.transpose(item_embeddings))
        