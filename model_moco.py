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
        self.base_model.fc2 = nn.Linear(1000, 256)
        self.classify = nn.Linear(256, 196)

    def forward(self, x):
        feature = self.base_model(x)
        feature = self.base_model.fc2(feature)
        feature = feature.view(feature.size(0), -1)
        pred = self.classify(feature)
        return pred, feature

