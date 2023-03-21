import os
import platform

if "mac" in platform.platform():
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import copy

from pipeline import *
from training import *
from mcts import *
from agents import *
from utils import *

if __name__ == "__main__":
    make_if_doesnt_exist("datasets")
    mp.set_start_method("spawn")
    net = Network().to(device=device, dtype=torch.float32)
    net.eval()

    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=0.5)
    criterion_p = nn.CrossEntropyLoss()
    criterion_v = nn.MSELoss()

    last_check, name = get_last_checkpoint()
    start_epoch = 0
    if last_check != None:
        print("Loading checkpoint at path", name)
        net.load_state_dict(last_check["network"])
        optimizer.load_state_dict(last_check["optimizer"])
        start_epoch = last_check["epoch"]

    for epoch in range(start_epoch + 1, epochs + 1):
        plist = []

        epoch_path, epoch_states, epoch_post, epoch_vals = prep_files_epoch(epoch)

        for w in range(workers):
            proc = mp.Process(
                target=run_game, 
                args=(copy.deepcopy(net).share_memory(), w, epoch_states, epoch_post, epoch_vals)
            )
            proc.start()
            plist.append(proc)
    
        for p in plist:
            p.join()

        states = read_all_data(epoch_states)
        post = read_all_data(epoch_post)
        vals = read_all_data(epoch_vals)

        loader = DataLoader(
            DataBuffer(states, post, vals),
            batch_size=batch_size,
            shuffle=True
        )

        train(loader, net, criterion_p, criterion_v, optimizer)
        save_model(net, epoch, optimizer)

        
    