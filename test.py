import torch
import torch.nn as nn
import torch.optim as optim
from agents import *
import random

import matplotlib.pyplot as plt

log_stride = 1
batch_size = 256

model = Network().cuda().share_memory()
model.train()

def unravel_index(
    indices: torch.LongTensor,
    shape
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord.flip(-1)

print(model(torch.zeros((1, 1, 7, 7), device="cuda")))
exit()

def generate_data():
    state = None
    post = None
    val = None
    if random.random() < 0.0:
        state = torch.zeros((batch_size, 7, 7), device="cuda", dtype=torch.float32)
        post = torch.randint(1, 2, (batch_size,))
        val = torch.ones((batch_size,)).tanh()
    else:
        state = torch.randn((batch_size, 49), device="cuda", dtype=torch.float32)
        state[state < 0] = 0
        post = state.argmax(dim=-1)
        state[state > 0] = 1
        state[post] = 1
        val = state.mean(dim=1)
        state = state.reshape(batch_size, 7, 7)
    return state, post.to(device="cuda"), val.to(device="cuda")

s, p, v = generate_data()

loss_p = nn.CrossEntropyLoss()
loss_v = nn.L1Loss(reduction="mean")
optimizer = optim.SGD(params=model.parameters(), lr=0.005, weight_decay=0.1)

fig, axs = plt.subplots(2)
axs[0].set_title("Prior loss")
axs[1].set_title("Value loss")

data_x = []
data_p = []
data_v = []
 
avp = 0
avv = 0
for epoch in range(1, 201):
    state, post, val = generate_data()
    #print(state.shape, post.shape, val.shape)
    prior, pred = model(state.view(-1, 1, 7, 7), view=False)
    
    optimizer.zero_grad()
    l1 = loss_p(prior, post)
    l2 = loss_v(pred, val)
    loss = l1 + l2
    loss.backward()
    optimizer.step()

    avp += l1.item() / log_stride
    avv += l2.item() / log_stride
    if epoch % log_stride == 0:
        print(f"Epoch {epoch}: Prior loss = {avp:.3f} | Value loss = {avv:.3f}")
        data_x.append(epoch)
        data_p.append(avp)
        data_v.append(avv)
        avp = 0
        avv = 0

axs[0].plot(data_x, data_p)
axs[1].plot(data_x, data_v)
plt.show()

test_state, test_post, test_val = generate_data()

print(" Testing ".center(50, "="))
with torch.inference_mode():
    prior, pred = model(state.view(-1, 1, 7, 7), view=False)
    l1 = loss_p(prior, test_post)
    l2 = loss_v(pred, test_val)
    print("Test loss l1", l1.item())
    print("Test loss l2", l2.item())
    print(test_post)
    print(test_val)
    print(prior)
    print(pred)