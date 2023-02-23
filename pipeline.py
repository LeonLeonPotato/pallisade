import time
import pickle

from training import *
from mcts import *
from agents import *
from utils import *

def run_game(network, worker_id, epoch):
    state_history = np.empty(0, dtype=np.float32)
    posterior_history = np.empty(0, dtype=np.float32)
    vals = np.empty(0, dtype=int)

    for g in range(self_play_games):
        node = prep_empty_board()

        his = 0
        while True:
            best, picked = predict(network, node)
            state_history.append(node.state * node.turn)
            posterior_history.append(node.children.index(best))
            his += 1

            node = picked if mcts_stochastic else best
            node.parent = None
            
            res = check_win(node.state)
            if res != 2:
                print("Won:", res)
                break
        
        if res == 0:
            tmp = np.zeros(his, dtype=int)
        else:
            tmp = alternating_tensor(his, res)
        vals = np.concatenate((vals, tmp))

        print("Worker {0} finished game {1} | winner: {2} | history: {3}"
              .format(worker_id, g, res, his))

    data = np.array(state_history, posterior_history, vals)
    path = os.path.join("datasets", str(epoch), f"data-{worker_id}")

    with open(path, "xb") as f:
        np.save(f, data)
    
    del data