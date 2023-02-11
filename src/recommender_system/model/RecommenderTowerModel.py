import torch.nn as nn

class RecommenderTowerModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dimension: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dimension)
        self.fc = nn.Linear(embedding_dimension,256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256,embedding_dimension)
        self.relu2 = nn.ReLU()
        #self.fc3=nn.Linear(128,embedding_dimension)
        #self.relu3=nn.ReLU()

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        #x = self.fc3(x)
        #x = self.relu3(x)
        return x
        