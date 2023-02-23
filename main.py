import os
import platform

if "mac" in platform.platform():
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from pipeline import *
from training import *
from mcts import *
from agents import *
from utils import *

if __name__ == "__main__":
    make_if_doesnt_exist("datasets")
    mp.set_start_method("spawn")

    net = Network().to(device=device).share_memory()

    optimizer = optim.Adam(params=net.parameters())
    criterion_p = nn.CrossEntropyLoss()
    criterion_v = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        plist = []

        epoch_path = os.path.join("datasets", f"dataset-{str(epoch)}")
        make_if_doesnt_exist(epoch_path)
        epoch_states = os.path.join(epoch_path, "state")
        make_if_doesnt_exist(epoch_states)
        epoch_post = os.path.join(epoch_path, "post")
        make_if_doesnt_exist(epoch_post)
        epoch_vals = os.path.join(epoch_path, "val")
        make_if_doesnt_exist(epoch_vals)

        for w in range(workers):
            proc = mp.Process(
                target=run_game, 
                args=(net, w, epoch_states, epoch_post, epoch_vals)
            )
            proc.start()
            plist.append(proc)
    
        for p in plist:
            p.join()

        states = None
        post = None
        vals = None

        for file in os.listdir(epoch_states):
            with open(os.path.join(epoch_states, file), "rb") as f:
                nparr = torch.from_numpy(np.load(f))
                if states == None:
                    states = nparr
                else:
                    states = torch.concat((states, nparr))
        for file in os.listdir(epoch_post):
            with open(os.path.join(epoch_post, file), "rb") as f:
                nparr = torch.from_numpy(np.load(f))
                if post == None:
                    post = nparr
                else:
                    post = torch.concat((post, nparr))
        for file in os.listdir(epoch_vals):
            with open(os.path.join(epoch_vals, file), "rb") as f:
                nparr = torch.from_numpy(np.load(f))
                if vals == None:
                    vals = nparr
                else:
                    vals = torch.concat((vals, nparr))

        loader = DataLoader(
            DataBuffer(states, post, vals),
            batch_size=batch_size,
            shuffle=True
        )

        train(loader, net, criterion_p, criterion_v, optimizer)
        

        
    