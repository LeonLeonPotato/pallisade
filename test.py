import torch.nn as nn
import torch.nn.functional as F
import torch

inp = torch.randn(10).unsqueeze(0).unsqueeze(0)
print(inp)

conv = nn.Conv1d(1, 1, 1, 1)
print(conv.weight, conv.bias)
print(conv(inp))