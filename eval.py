import torch
import agents
import utils
import numpy as np

def display(data):
    ret = np.chararray((7, 7))
    for x in range(7):
        for y in range(7):
            if data[x, y].item() == 1:
                ret[x, y] = 'X'
            elif data[x, y].item() == -1:
                ret[x, y] = "O"
            else:
                ret[x, y] = "`"
    return " " + str(ret).replace("b", "").replace('\'', "")[1:-1]

net = agents.Network()
net(torch.randn((1, 1, 7, 7)))

dic = torch.load("save-15", map_location=torch.device("cpu"))
net.load_state_dict(dic['network'])

board = torch.zeros(7, 7)

turn = -1
with torch.inference_mode():
    while True:
        print(display(board))
        won = utils.check_win(board.numpy())
        if won != 2:
            print("Won:", won)
            break
        if turn == 1:
            print("Input move: ", end='')
            y, x = tuple(map(int, input().split(" ")))
            x -= 1
            y -= 1
            if board[x, y] == 0:
                board[x, y] = 1.0
                turn *= -1
                print("Network eval:", net((board).unsqueeze(0).unsqueeze(0))[1].item())
            else:
                print("Bad move!")
        else:
            prior, val = net((board * -1).unsqueeze(0).unsqueeze(0))
            prior = prior.squeeze(0).reshape((49, ))
            prior[prior == 0] = -99999
            prior = prior.softmax(dim=-1)
            prior = prior.reshape((7, 7))
            flat = prior.argmax().item()
            x = flat // 7
            y = flat % 7
            board[x, y] = -1
            print(f"Confidence: {prior.max().item()} | Value: {val.item()}")
            turn *= -1