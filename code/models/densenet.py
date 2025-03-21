import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNet(nn.Module):
    def __init__(self, emb_dims):
        super().__init__()

        self.model = torch.hub.load('pytorch/vision:v0.8.0', 'densenet121', pretrained=True)
        # or any of these variants
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet169', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet161', pretrained=True)

        self.linear1 = nn.Linear(1000, emb_dims, bias=False)
        self.bn1 = nn.BatchNorm1d(emb_dims)
    
    def forward(self, x):
        x = self.model(x)
        x = F.relu(self.bn1(self.linear1(x)))
        return x
