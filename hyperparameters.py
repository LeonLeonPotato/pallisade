import torch
import sys

_init = False

epochs = 100
epochs_per_dataset = 10
self_play_games = 1
batch_size = 4
learning_rate = 0.001
workers = 1

max_tasks = 256
tasks_timeout = 0.1

mcts_dirichlet = 0.25
mcts_dirichlet_val = 0.03
mcts_cpuct_param = 2.0
mcts_searches = 49 * 2

device = None
colab_env = None

if not _init:
    _init = True
    colab_env = 'google.colab' in sys.modules
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    