import torch.nn as nn
import concurrent.futures
from collections import deque
import threading

from hyperparameters import *
from mcts import *

_HISTORY = 1
_FLAT = 7 * 7 * 8

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

class PolicyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=1)
        self.norm = nn.BatchNorm1d(1)
        self.fc = nn.Linear(_FLAT, 49)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.leaky_relu(x)
        x = self.fc(x)
        return x

class ValueHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=1)
        self.norm = nn.BatchNorm1d(1)
        self.fc1 = nn.Linear(_FLAT, 32)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.leaky_relu(x)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        return x

class Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=_HISTORY, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(8)
        self.trunk = nn.Sequential(
            *[ResidualLayer() for i in range(4)]
        )
        self.policy = PolicyHead()
        self.value = ValueHead()
    
    def forward(self, x, view=True): # x is shape (Batch, _HISTORY, N, M)
        tmp = self.conv(x)
        tmp = self.norm(tmp)  # tmp is shape (Batch, 8, N, M)
        tmp = self.trunk(tmp) # tmp is shape (Batch, 8, N, M)
        tmp = tmp.flatten(start_dim=1, end_dim=-1).unsqueeze(1) # tmp is shape (Batch, 1, N * M * 8)
        pol:torch.Tensor = self.policy(tmp) # tmp is shape (Batch, 49)
        val = self.value(tmp) # tmp is shape (Batch, 1)
        
        if view:
            pol = pol.reshape((-1, 7, 7))
            pol[x[:, -1, :, :] != 0] = 0
        else:
            pol = pol.squeeze(1)
            ex = x[:, -1, :, :].flatten(start_dim=1, end_dim=-1)
            pol[ex != 0] = 0

        return pol, val.flatten()

# class Manager():
#     def __init__(self, network) -> None:
#         self.locks = [threading.Lock() for i in range(workers)]
#         self.outs = [0 for i in range(workers)]
#         self.tasks = deque()
#         self.network = network

#     def submit(self, task, id):
#         self.tasks.append(task)
#         self.locks[id]
        
        
#     def start_loop(self):
#         while True:



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