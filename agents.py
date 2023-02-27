import torch.nn as nn

from hyperparameters import *
from mcts import *

from concurrent import futures
from threading import Lock, Thread

# Model parameters, not really "hyperparams"
_LAYERS = 5
_FILTERS = 64
_HISTORY = 1
_FLAT = 7 * 7 * 64

class GPUCache():
    def __init__(self, model) -> None:
        self.last = time.time()
        self.sumbitted = 0
        self.cur_lock = 0
        self.tasks = []
        self.out = None
        self.model = model
        self.finished = False

    def lock(self):
        self.locks = [Lock() for i in range(max_tasks)]
        for i in self.locks: 
            i.acquire()

    def run(self):
        self.lock()
        while True:
            time.sleep(0.05)
            if self.sumbitted >= max_tasks or (time.time() - self.last >= tasks_timeout and self.sumbitted > 0):
                self.flush()
            if self.finished:
                break

    def submit(self, task):
        # print(f"Submitted task at length {len(task)}")
        self.tasks.append(task)
        start = self.sumbitted
        self.sumbitted += len(task)
        self.cur_lock += 1
        return start, self.sumbitted, self.cur_lock - 1
    
    @torch.no_grad()
    def flush(self):
        # print(f"Flushing with size of {len(self.tasks)}")
        self.cur_lock = 0
        self.sumbitted = 0
        self.last = time.time()
        inp = torch.cat(self.tasks).to(device="cuda")
        self.out = self.model(inp)
        for lock in self.locks:
            lock.release()
        for lock in self.locks:
            lock.acquire()
        self.tasks.clear()
        # print("Out size =", len(self.out[0]))

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
        self.diri = torch.distributions.dirichlet.Dirichlet(torch.tensor([mcts_dirichlet_val] * 7 * 7))
    
    def forward(self, x, view=True): # x is shape (Batch, _HISTORY, N, M)
        tmp = self.conv(x)
        tmp = self.norm(tmp)  # tmp is shape (Batch, _FILTERS, N, M)
        tmp = self.trunk(tmp) # tmp is shape (Batch, _FILTERS, N, M)
        tmp = tmp.flatten(start_dim=1, end_dim=-1).unsqueeze(1) # tmp is shape (Batch, 1, N * M * _FILTERS)
        pol = self.policy(tmp) # tmp is shape (Batch, 49)
        val = self.value(tmp) # tmp is shape (Batch, 1)

        if not self.training:
            diri = self.diri.sample_n(x.shape[0]).to(device=device)
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
    print("childs", len(root.children))
    print(root.children_P)

    plist = []
    with futures.ThreadPoolExecutor(max_workers=12) as pool:
        for s in range(mcts_searches):
            plist.append(pool.submit(search, root, cache))

    futures.as_completed(plist)
    cache.finished = True
    cache_thread.join()

    for p in plist:
        if p.exception():
            import traceback
            exc = p.exception()
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            exit()
    print(root.state)
    print("root w", root.W)
    
    best = max(root.children, key=lambda x: x.uct())
    del cache
    return best