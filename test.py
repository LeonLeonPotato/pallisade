import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from threading import Thread, Lock
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
import random

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
        self.policy = nn.Linear(128 * 7 * 7, 7 * 7)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.trunk(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.policy(x)
        return x
    
model = Mod().cuda()

class Manager():
    def __init__(self) -> None:
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
            if self.cur_task > 100000 or (time.time() - self.last_shipped > 0.4 and self.cur_task > 0):
                oooo = time.time()
                self.append_lock.acquire()
                inp = torch.cat(self.tasks).cuda()
                out = model(inp)
                cur = 0
                for f, t in zip(self.futures, self.tasks):
                    f.set_result(out[cur:cur+len(t)])
                    cur += len(t)
                self.futures.clear()
                self.tasks.clear()
                print(f"Flushing with size of {self.cur_task}, {time.time() - self.last_shipped:.3f} seconds since last shipped.")
                self.cur_task = 0
                self.last_shipped = time.time()
                self.append_lock.release()
                print(time.time() - oooo)

manager = Manager()
manager_thread = Thread(target=manager.run)
manager_thread.start()

def work():
    try:
        time.sleep(random.random() * 0.1)
        batches = random.randint(20, 49)
        task = torch.randn((batches, 1, 7, 7), dtype=torch.float32)
        future = manager.submit(task)
        if future.result().shape[0] != task.shape[0]:
            print("Wrong shape!")
            exit()
    except Exception as f:
        logger.exception(f)
        exit()

with ThreadPoolExecutor(max_workers=49) as pool:
    for i in range(1000):
        pool.submit(work)

print("Finished")
manager.exit = True
manager_thread.join()