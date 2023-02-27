import torch.nn as nn
import torch.nn.functional as F

from hyperparameters import *
from mcts import *

from concurrent import futures
from concurrent.futures import Future
from threading import Lock, Thread

# Model parameters, not really "hyperparams"
_LAYERS = 5
_FILTERS = 64
_HISTORY = 1
_FLAT = 7 * 7 * 64

class GPUCache():
    def __init__(self, model) -> None:
        self.model = model
        self.tasks = []
        self.futures = []
        self.cur_task = 0
        self.append_lock = Lock()
        self.last_shipped = time.time()
        self.exit = False

    def submit(self, task) -> Future:
        self.append_lock.acquire()
        ret = Future()
        self.futures.append(ret)
        self.tasks.append(task)
        self.cur_task += len(task)
        self.append_lock.release()
        return ret
    
    @torch.inference_mode()
    def run(self):
        while True:
            if self.exit:
                break
            time.sleep(0.05)
            self.append_lock.acquire()
            if self.cur_task > 512 or (time.time() - self.last_shipped > 0.15 and self.cur_task > 0):
                inp = torch.cat(self.tasks)
                out = self.model(inp)
                cur = 0
                for f, t in zip(self.futures, self.tasks):
                    prior = out[0][cur:cur+len(t)]
                    val = out[1][cur:cur+len(t)]
                    f.set_result((prior, val))
                    cur += len(t)
                self.futures.clear()
                self.tasks.clear()
                self.cur_task = 0
                self.last_shipped = time.time()
            self.append_lock.release()


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
        self.fc = nn.Linear(_FLAT, 49)
    
    def forward(self, x):
        x = self.fc(x)
        return x

class ValueHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(_FLAT, 32)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
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
        self.diri = torch.distributions.dirichlet.Dirichlet(torch.tensor([mcts_dirichlet_val] * 7 * 7))
    
    def forward(self, x, view=True): # x is shape (Batch, _HISTORY, N, M)
        tmp = self.conv(x)
        tmp = self.norm(tmp)  # tmp is shape (Batch, _FILTERS, N, M)
        tmp = self.trunk(tmp) # tmp is shape (Batch, _FILTERS, N, M)
        tmp = tmp.flatten(start_dim=1, end_dim=-1).unsqueeze(1) # tmp is shape (Batch, 1, N * M * _FILTERS)
        pol = self.policy(tmp) # tmp is shape (Batch, 49)
        val = self.value(tmp) # tmp is shape (Batch, 1)

        if not self.training:
            diri = self.diri.sample((x.shape[0], )).to(device=device)
            diri = diri.view(x.shape[0], 1, 49)
            pol = (1 - mcts_dirichlet) * pol + mcts_dirichlet * diri
        
        if view:
            pol = pol.reshape((-1, 7, 7))
            pol[x[:, -1, :, :] != 0] = 0
        else:
            pol = pol.squeeze(1)
            ex = x[:, -1, :, :].flatten(start_dim=1, end_dim=-1)
            pol[ex != 0] = 0

        return pol, val.flatten()

def predict(network, root : Node):
    cache = GPUCache(network)
    cache_thread = Thread(target=cache.run)
    cache_thread.start()

    root.expand(cache)
    childs = len(root.children)

    plist = []
    with futures.ThreadPoolExecutor(max_workers=49) as pool:
        for s in range(mcts_searches):
            plist.append(pool.submit(search, root.children[s % childs], cache))

    futures.as_completed(plist)
    cache.exit = True
    cache_thread.join()
    
    best = max(root.children, key=lambda x: x.uct())
    del cache

    torch.cuda.empty_cache()
    return best