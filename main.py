import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import os.path as path
import subprocess
import time
import sys

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from training import *
from mcts import *
from agents import *
from utils import *

net = Network().to(device=device)
optimizer = optim.Adam(params=net.parameters())
criterion_p = nn.CrossEntropyLoss()
criterion_v = nn.MSELoss()

agent1 = Agent(net)
agent2 = Agent(net)

playing = agent2

state_history = []
posterior_history = [] 

import os, psutil
process = psutil.Process(os.getpid())

for i in range(1, 101):
    node, board = prep_empty_board(net)

    print("Epoch", i)
    while True:
        print(process.memory_info().rss / 10e9)  # in bytes 
        playing = agent2 if playing == agent1 else agent1

        best, picked = playing.predict(node)
        state_history.append(node.state * node.turn)
        posterior_history.append(node.children.index(best))

        #print([f"{c.uct():.2f} {c.move}" for c in node.children])
        node = best
        node.parent = None
        print("Top chosen UCT:", node.uct())
        print("Turn:", node.turn)
        print("Won:", check_win(node.state))
        print(node.state)
        print("==============")
        
        res = check_win(node.state)
        if res != 2:
            break
    
    if res == 0:
        vals = torch.zeros(len(state_history), dtype=torch.float32)
    else:
        vals = alternating_tensor(len(state_history), res)

    buff = DataBuffer(
        torch.from_numpy(np.array(state_history, dtype=np.float32)).unsqueeze(1),
        torch.tensor(posterior_history, dtype=torch.long),
        res
    )

    train(
        DataLoader(buff, batch_size=batch_size, shuffle=True), 
        net, 
        criterion_p,
        criterion_v,
        optimizer    
    )

    state_history = []
    posterior_history = []

exit()