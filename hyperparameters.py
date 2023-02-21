import torch
import numpy as np
import sys

_init = False

batch_size = 4
learning_rate = 0.001

mcts_multinomial = True
mcts_top_p = 0.5
mcts_cpuct_param = 2.0
mcts_searches = 49

device = None
colab_env = None

if not _init:
    _init = True
    #torch.manual_seed(3)
    #np.random.seed(1)
    colab_env = 'google.colab' in sys.modules
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # we assume cuda lol (chad move)
    print("Using device:", device)
    print("In colab:", colab_env)