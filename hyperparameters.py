import torch
import sys

_init = False

epochs = 100
epochs_per_dataset = 10
self_play_games = 2
batch_size = 16
learning_rate = 0.003
workers = 4

mcts_dirichlet = 0.25
mcts_top_p = 0.5
mcts_cpuct_param = 2.0
mcts_searches = 49

device = None
colab_env = None

if not _init:
    _init = True
    colab_env = 'google.colab' in sys.modules
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    