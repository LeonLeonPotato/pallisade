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
    print(" Training ".center(50, "="))
    network.train()
    for i in range(epochs_per_dataset):
        total_prior_loss = 0
        total_vals_loss = 0
        times = 0
        for state, p, v in data:
            prior, val = network(state.unsqueeze(1), view=False)
            
            optimizer.zero_grad()
            loss_prior = criterion_p(prior, p)
            loss_val = criterion_v(val, v)
            loss = loss_prior + loss_val
            total_prior_loss += loss_prior.item()
            total_vals_loss += loss_val.item()
            times += 1
            loss.backward()
            optimizer.step()
        print(f"Training epoch {i} | Average prior loss: {total_prior_loss / times:.3f} | Average val loss: {total_vals_loss / times:.3f}")
    network.eval()
    print(" Collecting Data ".center(50, "="))