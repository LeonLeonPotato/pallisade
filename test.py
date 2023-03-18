import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

import matplotlib.pyplot as plt

class ResidualLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(8)
        self.norm2 = nn.BatchNorm2d(8)
    
    def forward(self, x):
        tmp = self.conv(x)
        tmp = self.norm(x)
        tmp = F.leaky_relu(tmp)
        tmp = self.conv2(x)
        tmp = self.norm2(x)
        tmp += x
        tmp = F.leaky_relu(tmp)
        return tmp

class Network(nn.Module):
    def __init__(self):
        super(self, nn.Module).__init__()
        
        self.pre_process = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=8)
        )

        self.trunk1 = nn.Sequential(
            *[ResidualLayer() for i in range(5)]
        )
        self.trunk2 = nn.Sequential(
            *[ResidualLayer() for i in range(5)]
        )
        self.trunk3 = nn.Sequential(
            *[ResidualLayer() for i in range(5)]
        )

        self.health = nn.Linear(1, 64)
        self.downsampler = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=5, padding=1, bias=False),
            nn.BatchNorm2d(num_features=8)
        )

    def forward(self, x):
        # x is (B, H, 5N, 5M)
        x = self.pre_process(x)

        pass


N, M = 5, 6
t1 = torch.randn((1, 1, 5*N, 5*M))
c1 = nn.Sequential(
    nn.Conv2d(1, 1, (5, 5), 5, 1),
)

print(c1(t1).shape)

model = Network().cuda().share_memory()
model.train()