import numpy as np
import time
import colorama as C

turn = -1
post = np.load("data-1")

def display(data):
    ret = np.chararray((7, 7))
    for x in range(7):
        for y in range(7):
            if data[x, y].item() == 1:
                ret[x, y] = 'X'
            elif data[x, y].item() == -1:
                ret[x, y] = "O"
            else:
                ret[x, y] = "_"
    return ret

for i, v in enumerate(np.load("data-0")):
    k = np.unravel_index(post[i], v.shape)
    v = display(v * turn)

    if v[k] == b"_":
        v[k] = "i"
    v = " " + str(v).replace("b", "").replace('\'', "")[1:-1]
    print(v)
    turn *= -1

    print("")