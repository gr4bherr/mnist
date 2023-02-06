#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 4

traindata = datasets.MNIST('mnist_data', train=True, download=True, transform=transforms.ToTensor())
testdata = datasets.MNIST('mnist_data', train=False, download=True, transform=transforms.ToTensor())

trainload = DataLoader(traindata, batch_size=BATCH_SIZE) 
testload = DataLoader(testdata, batch_size=BATCH_SIZE) 

# linear -> relu -> linear -> softmax
class ModelClass(nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = nn.Linear(28 * 28, 64)
    self.l2 = nn.Linear(64, 10)

  def forward(self, x):
    x = x.view(-1, 28 * 28)
    x = F.relu(self.l1(x))
    x = self.l2(x)
    return F.log_softmax(x, dim=1)

model = ModelClass()
print(model)

optimizer = optim.Adam(model.parameters(), lr=LR)

# ** TRAIN **
b = 0
for epoch in range(EPOCHS):
  for batch in trainload:
    X, y = batch # pixels, labels
    predict = model(X.view(-1, 28 * 28)) # flatten image + apply model (list of predictions)
    loss = F.nll_loss(predict, y) # cross entropy loss
    loss.backward() # calc gradient (back prop)
    optimizer.step() # update gradient
    optimizer.zero_grad() # 
    #print(model.state_dict())
    if b % 1000 == 0: print(f"batch {b:<5} loss: {loss.data:.03f}")
    b += 1

torch.save(model.state_dict(), "model.pt")

# ** TEST ** 
correct, total = 0, 0
with torch.no_grad():
    for batch in testload:
        X, y = batch
        predict = model(X.view(-1, 28 * 28)) 
        for idx, i in enumerate(predict):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
print(f"accuracy: {100 * correct / total:.02f}")
