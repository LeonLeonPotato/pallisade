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
        cur = time.time()
        node = prep_empty_board(network)
        print("Prepare board time:", time.time() - cur)

        his = 0
        while True:
            best = predict(network, node)

            tmp_state = node.state * node.turn
            tmp_post = np.ravel_multi_index(best.move, tmp_state.shape)
            state_history.append(tmp_state)
            posterior_history.append(tmp_post)
            his += 1

            node = best
            node.parent = None
            
            res = check_win(node.state)
            if res != 2:
                break
            print(f"Finished move {his}")

        vals.extend([res * -((i % 2) * 2 - 1) for i in range(1, his+1)])

        print("Worker {0} finished game {1} | winner: {2} | history: {3}"
              .format(worker_id, g, res, his))

    epoch_state = os.path.join(epoch_state, f"data-{worker_id}")
    epoch_post = os.path.join(epoch_post, f"data-{worker_id}")
    epoch_val = os.path.join(epoch_val, f"data-{worker_id}")

    with open(epoch_state, "wb") as f:
        np.save(f, np.array(state_history, dtype=np.float32))
    with open(epoch_post, "wb") as f:
        np.save(f, np.array(posterior_history, dtype=np.longlong))
    with open(epoch_val, "wb") as f:
        np.save(f, np.array(vals, dtype=np.float32) * 0.9999)

def save_model(network:nn.Module, epoch, optim:nn.Module):
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    d = {
        "epoch" : epoch,
        "network": network.state_dict(),
        "optimizer": optim.state_dict()
    }

    torch.save(d, os.path.join("checkpoints", f"save-{epoch}"))

def get_last_checkpoint():
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
        return None, None
    if len(os.listdir("checkpoints")) == 0:
        return None, None
    
    f = max(os.listdir("checkpoints"), key=lambda u: int(u.split('-')[-1]))
    return torch.load(os.path.join("checkpoints", f)), os.path.join("checkpoints", f)
