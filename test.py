import numpy as np

turn = -1
val = np.load("data-0")

for i in val:
    print(i * turn + 1 - 1)
    turn *= -1
    print("")