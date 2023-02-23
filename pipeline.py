import time
import pickle

from training import *
from mcts import *
from agents import *
from utils import *

def run_game(network, worker_id, epoch_state, epoch_post, epoch_val):
    state_history = []
    posterior_history = []
    vals = []

    for g in range(self_play_games):
        node = prep_empty_board(network)

        his = 0
        while True:
            best, picked = predict(network, node)
            tmp_state = node.state * node.turn
            tmp_post = node.children.index(best)
            state_history.append(tmp_state)
            posterior_history.append(tmp_post)
            his += 1

            node = picked if mcts_stochastic else best
            node.parent = None
            
            res = check_win(node.state)
            if res != 2:
                break
        
        if res == 0:
            tmp = np.zeros(his, dtype=np.float32)
        else:
            tmp = alternating_tensor(his, res)
        vals = np.concatenate((vals, tmp), dtype=np.float32)

        print("Worker {0} finished game {1} | winner: {2} | history: {3}"
              .format(worker_id, g, res, his))

    # print(his)
    # print(vals.shape)
    # print(posterior_history.shape)
    # print(state_history.shape)

    epoch_state = os.path.join(epoch_state, f"data-{worker_id}")
    epoch_post = os.path.join(epoch_post, f"data-{worker_id}")
    epoch_val = os.path.join(epoch_val, f"data-{worker_id}")

    with open(epoch_state, "wb") as f:
        np.save(f, np.array(state_history, dtype=np.float32))
    with open(epoch_post, "wb") as f:
        np.save(f, np.array(posterior_history, dtype=np.longlong))
    with open(epoch_val, "wb") as f:
        np.save(f, vals)