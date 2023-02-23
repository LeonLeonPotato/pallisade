import numpy as np

s = np.array([1, 5, 3, 2, 4])

with open("temp", "wb") as f:
    np.save(f, s)

with open("temp", "rb") as f:
    print(type(np.load(f)))
