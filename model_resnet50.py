import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.base_model.fc2 = nn.Linear(1000, 196)

    def forward(self, x):
        x = self.base_model(x)
        x = self.base_model.fc2(x)
        return x

