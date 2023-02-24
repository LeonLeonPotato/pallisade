import torch.nn as nn

from hyperparameters import *
from mcts import *

# Model parameters, not really "hyperparams"
_LAYERS = 5
_FILTERS = 64
_HISTORY = 1
_FLAT = 7 * 7 * 64

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

class PolicyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([1.0], dtype=torch.float32).to(device=device), requires_grad=True)
        self.b = nn.Parameter(torch.tensor([0.1], dtype=torch.float32).to(device=device), requires_grad=True)
        self.register_parameter("policy_wc", self.w)
        self.register_parameter("policy_bc", self.b)
        self.norm = nn.BatchNorm1d(1)
        self.fc = nn.Linear(_FLAT, 49)
    
    def forward(self, x):
        x = x * self.w + self.b
        x = self.norm(x)
        x = F.leaky_relu(x)
        x = self.fc(x)
        return x

class ValueHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([1.0], dtype=torch.float32).to(device=device), requires_grad=True)
        self.b = nn.Parameter(torch.tensor([0.1], dtype=torch.float32).to(device=device), requires_grad=True)
        self.register_parameter("value_wc", self.w)
        self.register_parameter("value_bc", self.b)
        self.norm = nn.BatchNorm1d(1)
        self.fc1 = nn.Linear(_FLAT, 32)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = x * self.w + self.b
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
        self.conv = nn.Conv2d(in_channels=_HISTORY, out_channels=_FILTERS, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(_FILTERS)
        self.trunk = nn.Sequential(
            *[ResidualLayer() for i in range(_LAYERS)]
        )
        self.policy = PolicyHead()
        self.value = ValueHead()
    
    def forward(self, x, view=True): # x is shape (Batch, _HISTORY, N, M)
        tmp = self.conv(x)
        tmp = self.norm(tmp)  # tmp is shape (Batch, _FILTERS, N, M)
        tmp = self.trunk(tmp) # tmp is shape (Batch, _FILTERS, N, M)
        tmp = tmp.flatten(start_dim=1, end_dim=-1).unsqueeze(1) # tmp is shape (Batch, 1, N * M * _FILTERS)
        pol = self.policy(tmp) # tmp is shape (Batch, 49)
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


def predict(network, root : Node):
    root.expand(network)
    l = len(root.children)

    for s in range(mcts_searches):
        search(network, root.children[s % l])

    probs = torch.tensor([u.uct() for u in root.children]).softmax(dim=-1).numpy()
    indicies = np.random.multinomial(1, probs).argmax()
    best = root.children[probs.argmax().item()]
    picked = root.children[indicies.item()]
    return best, picked