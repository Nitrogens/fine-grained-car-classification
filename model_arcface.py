from __future__ import print_function
from __future__ import division

import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = self.alpha * (1 - p) ** self.gamma * logp
        return loss.mean()

class ArcFaceMetric(nn.Module):
    def __init__(self, dim_in, dim_out, s=30.0, m=0.50, easy_margin=False):
        super(ArcFaceMetric, self).__init__()
        
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.eps = 1e-12
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
        self.layer = torch.nn.Parameter(torch.FloatTensor(dim_out, dim_in))
        
        self._initialize_weight()

    def _initialize_weight(self):
        nn.init.xavier_uniform_(self.layer)

    def forward(self, x, y, device, test=False):
        cosine = F.linear(F.normalize(x), F.normalize(self.layer)).clamp(-1. + self.eps, 1. - self.eps)
        if test:
            return cosine
        sine = torch.sqrt((1.000000001 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)    # From label to one-hot vectors
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output

class model(nn.Module):
    def __init__(self, s=30.0, m=0.50, easy_margin=False):
        super(model, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.base_model.fc2 = nn.Linear(1000, 256)
        self.arcface = ArcFaceMetric(256, 196, s, m, easy_margin)

    def forward(self, x, y, device, test=False):
        x = self.base_model(x)
        x = self.base_model.fc2(x)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.arcface(x, y, device, test)
        return x

