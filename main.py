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
criterion = nn.CrossEntropyLoss()

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

        print([f"{c.uct():.2f} {c.debug}" for c in node.children])
        node = best
        print(node.uct())
        print("turn", node.turn)
        print(node.state)
        print("Won:", check_win(node.state))
        print("==============")
        
        res = check_win(node.state)
        if res != 0:
            break
    
    buff = DataBuffer(
        torch.from_numpy(np.array(state_history, dtype=np.float32)),
        torch.tensor(posterior_history),
        torch.fill(torch.zeros(len(state_history), dtype=torch.int), [0, 0, 2, 1][res])
    )

    train(
        DataLoader(buff, batch_size=batch_size, shuffle=True), 
        net, 
        criterion,
        optimizer    
    )

    state_history = []
    posterior_history = []

exit()