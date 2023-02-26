import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt

_FILTERS = 128

class ResidualLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=_FILTERS, out_channels=_FILTERS, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=_FILTERS, out_channels=_FILTERS, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(_FILTERS)
        self.norm2 = nn.BatchNorm2d(_FILTERS)
    
    def forward(self, x):
        tmp = self.conv(x)
        tmp = self.norm(x)
        tmp = F.leaky_relu(tmp)
        tmp = self.conv2(x)
        tmp = self.norm2(x)
        tmp += x
        tmp = F.leaky_relu(tmp)
        return tmp
    
class Mod(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            *[ResidualLayer() for i in range(4)]
        )
        self.conv = nn.Conv2d(1, _FILTERS, 3, 1, 1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.trunk(x)
        return x
    
test_mod = Mod().to(device="cuda").share_memory()

test_batch = torch.randn((120, 1, 7, 7), device="cuda")
test_mod(test_batch)

x = []
y = []
tests = 100
stride = 10

@torch.no_grad()
def test():
    for t in range(10, tests * stride + 1, stride):
        test_batch = torch.randn((t, 1, 7, 7), device="cuda")

        cur = time.time()
        sample = test_mod(test_batch)[0][0][0][0]
        passed = time.time() - cur
    
        print(f"Test {t} | Sample: {sample:.2f} | Time: {passed} seconds")

        x.append(t)
        y.append(passed)

        torch.cuda.empty_cache()

test()

plt.plot(x, y, "P", color="red")
plt.show()