import time
import torch.nn as nn
import torch.nn.functional as F

from hyperparameters import *
from mcts import *

from concurrent import futures
from concurrent.futures import Future
from threading import Lock, Thread

# Model parameters, not really "hyperparams"
_LAYERS = 8
_FILTERS = 8
_HISTORY = 1
_FLAT = 7 * 7 * _FILTERS

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
    
    @torch.no_grad()
    def run(self):
        while True:
            if self.exit:
                break
            time.sleep(0.05)
            self.append_lock.acquire()
            if self.cur_task > max_tasks or (time.time() - self.last_shipped > 0.15 and self.cur_task > 0):
                inp = torch.cat(self.tasks).to(device=device)
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
        self.fc1 = nn.Linear(_FLAT, 128)
        self.fc2 = nn.Linear(128, 49)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = x.reshape((-1, 7, 7))
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
    
    def forward(self, x, view=True): # x is shape (Batch, _HISTORY, N, M)
        tmp = self.conv(x)
        tmp = self.norm(tmp)  # tmp is shape (Batch, _FILTERS, N, M)
        tmp = self.trunk(tmp) # tmp is shape (Batch, _FILTERS, N, M)
        tmp = tmp.flatten(start_dim=1, end_dim=-1) # tmp is shape (Batch, N * M * _FILTERS)
        pol = self.policy(tmp) # tmp is shape (Batch, 7, 7)
        val = self.value(tmp) # tmp is shape (Batch, 1)

        mask = x[:, -1, :, :] != 0
        pol[mask] = 0.0

        if not view:
            pol = pol.reshape((-1, 49))
        
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