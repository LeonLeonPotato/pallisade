import torch.nn as nn
import concurrent.futures

from hyperparameters import *
from mcts import *

_FLAT = 7 * 7 * 1

class ResidualLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(1)
        self.norm2 = nn.BatchNorm2d(1)
    
    def forward(self, x):
        tmp = self.conv(x)
        tmp = self.norm(x)
        tmp = F.leaky_relu(tmp)
        tmp = self.conv2(x)
        tmp = self.norm2(x)
        tmp += x
        tmp = F.leaky_relu(tmp)
        return tmp

class PolicyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear()
    
    def forward(self, x):
        return self.fc(x)

class ValueHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(_FLAT, 32)
        self.fc2 = nn.Linear(_FLAT, 32)
    
    def forward(self, x):
        return self.fc(x)

class Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            *[ResidualLayer() for i in range(4)]
        )
        self.policy = PolicyHead()
        self.value = ValueHead()
    
    def forward(self, x, view=True): # x is shape (Batch, 1, N, M)
        tmp:torch.Tensor = self.trunk(x)
        tmp = tmp.flatten(start_dim=1, end_dim=-1)
        pol = self.policy(tmp)
        val = self.value(tmp)
        
        if view:
            pol = pol.view(7, 7)
            pol[x != 0] = 0
        else:
            pol[x.flatten() != 0] = 0

        return pol, val

class Agent():
    def __init__(self, network : Network) -> None:
        self.network = network

    def predict(self, root : Node):
        root.expand(self.network)
        l = len(root.children)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            for s in range(mcts_searches):
                futures.append(executor.submit(search, self.network, root.children[s % l]))
            concurrent.futures.wait(futures)

        probs = torch.tensor([u.uct() for u in root.children]).softmax(dim=-1).numpy()
        indicies = np.random.multinomial(1, probs).argmax()
        best = root.children[probs.argmax().item()]
        picked = root.children[indicies.item()]
        return best, picked