import torch
import torch.nn as nn
import torch.optim as optim
from agents import *

import matplotlib.pyplot as plt

log_stride = 10

model = Network()
model.train()

def generate_data():
    state = torch.randn((64, 1, 49))
    state[state < 0.2] = 0
    post = (state ** 2 - state + 2).argmax(-1).T.squeeze()
    val = state.mean(dim=-1).T.squeeze().tanh()
    return state, post, val

s, p, v = generate_data()

loss_p = nn.CrossEntropyLoss()
loss_v = nn.SmoothL1Loss(reduction="mean")
optimizer = optim.Adam(params=model.parameters(), lr=0.005, weight_decay=0.01)

fig, axs = plt.subplots(2)
axs[0].set_title("Prior loss")
axs[1].set_title("Value loss")

data_x = []
data_p = []
data_v = []
 
avp = 0
avv = 0
for epoch in range(1, 751):
    state, post, val = generate_data()
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