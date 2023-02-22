import os
import os.path as path
import subprocess
import time
import sys
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperparameters import *
from utils import *

class Node():
    def __init__(self, state, parent, turn) -> None:
        self.leaf = True
        self.Q = 0 # stored total Q value
        self.P = 0 # stored prior probability
        self.W = 0
        self.visits = 0 # stored number of visits to this node
        self.children_P = None
        self.children = [] # child nodes
        self.parent = parent
        self.state = state
        self.turn = turn # self = turn, children = -turn
        self.move = None
    
    def expand(self, network):
        if self.leaf:
            # prior, value = network(self.state)
            self.leaf = False
            batched = []
            for x, y in get_possible_actions(self.state):
                tmp = self.state.copy()
                new_node = Node(tmp, self, -self.turn) # create new node
                new_node.state[x, y] = new_node.turn # apply action
                new_node.move = [x, y]

                batched.append(torch.tensor(new_node.state * new_node.turn, dtype=torch.float32, device=device).unsqueeze(0))
                new_node.P = self.children_P[x, y].item()
                self.children.append(new_node)

            with torch.no_grad():
                inp = torch.stack(batched)
                child_prior, q_vals = network(inp)

            for i in range(len(self.children)):
                self.children[i].children_P = child_prior[i]
                self.children[i].Q = q_vals[i].item()
                self.backprop(new_node.Q)
    
    def pick_best_move(self):
        max_child = max(self.children, key=lambda k: k.uct())
        return max_child

    def backprop(self, value):
        current = self
        while True:
            current.W += 1
            current.Q += value * -current.turn
            current = current.parent
            if current == None:
                break

    def uct(self):
        if (self.parent == None):
            return self.P / (1 + self.visits)
    
        u = self.P * math.sqrt(self.parent.visits) / (1 + self.visits)
        q = self.Q / max(1, self.W)
        return q + u
    
    def __str__(self):
        return str(self.state)

def prep_empty_board(network):
    board = np.zeros((7, 7), dtype=int)
    node = Node(board, None, -1)
    with torch.no_grad():
        priors, value = network(torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0))
        node.children_P = priors[0]
        node.Q = value[0].item()
    return node, board

def search(net, root:Node):
    while True:
        res = check_win(root.state)
        if res != 2:
            root.backprop(abs(res))
            break
        root.expand(net)
        root = root.pick_best_move()
        root.visits += 1