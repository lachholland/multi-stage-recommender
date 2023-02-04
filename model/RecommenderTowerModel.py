import torch.nn as nn

class RecommenderTowerModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dimension: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dimension)
        self.fc = nn.Linear(256, 120)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(embedding_dimension)
        self.relu2 = nn.ReLU()
        
    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x