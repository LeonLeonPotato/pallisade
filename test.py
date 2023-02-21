import torch.nn as nn
import torch.nn.functional as F
import torch


conv = nn.Conv2d(1, 1, 3, 1, 1)
conv2 = nn.Conv2d(1, 1, 3, 1, 1)
norm = nn.BatchNorm2d(1)
norm2 = nn.BatchNorm2d(1)

inp = torch.randn((3, 3))
inp = inp.unsqueeze(0).unsqueeze(1)

res = norm(conv(inp))
res = F.leaky_relu(res)
res = norm2(conv2(res)) + inp
res = F.leaky_relu(res)

print(inp)
print(res)