import os
import platform

if "mac" in platform.platform():
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from pipeline import *
from training import *
from mcts import *
from agents import *
from utils import *

if __name__ == "__main__":
    if not os.path.exists("datasets"):
        os.mkdir("datasets")
    
    mp.set_start_method("spawn")

    net = Network().to(device=device).share_memory()

    optimizer = optim.Adam(params=net.parameters())
    criterion_p = nn.CrossEntropyLoss()
    criterion_v = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        plist = []
        for w in range(workers):
            proc = mp.Process(
                target=run_game, 
                args=(net, w, epoch)
            )
            proc.start()
            plist.append(proc)
    
        for p in plist:
            p.join()

        epoch_path = os.path.join("dataset", str(epoch))

        data = np.empty(0)

        for file in os.listdir(epoch_path):
            with open(os.path.join(epoch_path, file), "rb") as f:
                torch.concat((data, np.load(f)))

        
    