import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from hyperparameters import *
from mcts import *

class DataBuffer(Dataset):
    def __init__(self, state, mcts_results, val) -> None:
        super().__init__()
        assert len(state) == len(mcts_results) == len(val)

        self.states = state
        self.mcts = mcts_results
        self.val = val
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, index):
        return self.states[index].to(device=device), self.mcts[index].to(device=device), self.val[index].to(device=device)
    
def train(data:DataLoader, network, criterion_p, criterion_v, optimizer):
    for state, p, v in data:
        prior, val = network(state, view=False)
        
        optimizer.zero_grad()
        loss_prior = criterion_p(prior, p)
        loss_val = criterion_v(val, v)
        loss = loss_prior + loss_val
        print(loss.item())
        loss.backward()
        optimizer.step()