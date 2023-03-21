import numpy as np
import torch
from threading import Lock

from hyperparameters import *
from utils import *

diri = torch.distributions.dirichlet.Dirichlet(torch.tensor([mcts_dirichlet_val] * 7 * 7))

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
        self.access = Lock()
    
    def expand(self, cache):
        self.access.acquire()
        if self.leaf:
            # prior, value = network(self.state)
            self.leaf = False
            batched = []
            for x, y in get_possible_actions(self.state):
                tmp = self.state.copy()
                new_node = Node(tmp, self, -self.turn) # create new node
                new_node.state[x, y] = new_node.turn # apply action
                new_node.move = [x, y]

                batched.append(torch.tensor(new_node.state * new_node.turn, dtype=torch.float32).unsqueeze(0))
                new_node.P = self.children_P[x, y].item()
                self.children.append(new_node)

            inp = torch.stack(batched)
            future = cache.submit(inp)
            child_prior, q_vals = future.result()
            child_prior = child_prior.reshape((-1, 49))
            child_prior[child_prior == 0] = -9999
            noise = diri.sample((len(child_prior), )).to(device=device)
            child_prior = (1 - mcts_dirichlet) * child_prior + mcts_dirichlet * noise
            child_prior = child_prior.softmax(dim=-1)
            child_prior = child_prior.reshape((-1, 7, 7))
            for i in range(len(self.children)):
                child = self.children[i]
                child.children_P = child_prior[i]
                child.Q = q_vals[i].item()
                self.backprop(child.Q)
        self.access.release()

    
    def pick_best_move(self):
        max_child = max(self.children, key=lambda k: k.uct())
        return max_child

    def backprop(self, value):
        current = self
        turn = -1
        while True:
            current.W += 1
            current.Q += value * turn
            current = current.parent
            turn *= -1
            if current == None:
                break

    def uct(self):
        if (self.parent == None):
            v = 1 + self.visits
        else:
            v = self.parent.visits
    
        u = self.P * v / (1 + self.visits)
        q = self.Q / max(1, self.W)
        return q + u
    
    def __str__(self):
        return str(self.state)

def prep_empty_board(network):
    board = np.zeros((7, 7), dtype=np.int8)
    node = Node(board, None, -1)
    with torch.no_grad():
        priors, value = network(torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0))
        node.children_P = priors[0].reshape((49,)).softmax(dim=-1).reshape((7, 7))
        node.Q = value[0].item()
    return node

def search(root:Node, cache):
    his = 0
    while True:
        res = check_win(root.state)
        if res != 2:
            root.backprop(-1)
            break
        root.expand(cache)
        root = root.pick_best_move()
        root.visits += 1
        his += 1