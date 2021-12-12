
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class ClsModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.fc = nn.Sequential(
            nn.Linear(1280, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, num_classes, bias=True)
        )
    def forward(self, x):
        x = self.model.extract_features(x)
        x = self.model._avg_pooling(x)
        
        x = x.flatten(start_dim=1)
        x = self.model._dropout(x)
        x = self.fc(x)
        return x
