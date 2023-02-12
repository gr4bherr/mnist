#!/usr/bin/env python3
import numpy as np

def displayImage(num):
  print(labels[num])
  for i in range(num, len(images)):
    print(f"{images[i][num]:.2f}", end=" ")
    if (i+1) % 28 == 0:
      print()

with open('train.npy', 'rb') as f:
  images = np.load(f)
  labels = np.load(f)

imgnum, pxlnum = images.shape # todo: rename to batch size or X (Xtrain, Xtest)

EPOCHS = 1
BS = 25
LR = 1e-3

# negative gradient of cost function = avg changes we want to make (to weights & biases) over all images

def sigmoid(z):
  return 1.0 / (1.0 + np.exp(z))

def sigmoid_der(z):
  return sigmoid(z) - (1 - sigmoid(z))

def squarederror(x, y):
  return (x - y) ** 2

def squarederror_der(x, y):
  return 2 * (x - y) 


class layer:
  def __init__(self, f, t):
    self.weights = np.random.rand(t, f) - 0.5
    self.biases = np.random.rand(t, 1) - 0.5
    self.dfrom = f # dim from
    self.dto = t # dim to
  
  def z(self, x): # todo: better name
    return self.weights @ x - self.biases
  

# todo: editable only init and forward, rest automatic, inherit
class model:
  def __init__(self, *dims):
    l0 = None
    l1 = layer(784, 64)
    l2 = layer(64, 16)
    l3 = layer(16, 10)
    self.layers = [l0, l1, l2, l3]
    self.dimnum = len(self.layers)
  
  def forward(self, x, y):
    self.y = y
    z0 = None
    a0 = x
    z1 = self.layers[1].z(a0)
    a1 = sigmoid(z1)
    z2 = self.layers[2].z(a1)
    a2 = sigmoid(z2)
    z3 = self.layers[3].z(a2)
    a3 = sigmoid(z3)
    self.activations = [a0, a1, a2, a3]
    self.lincomb = [z0, z1, z2, z3]
  
  # *** BACK ***
  def _endgrad(self, L):
    # go through all nodes in next layer
    if L == self.dimnum - 1: # last layer
      # loss function derivative
      return squarederror_der(self.activations[L][:, self.currentbatch], self.y[:, self.currentbatch])
    else:
      if self.doweight:
        a = self.layers[L+1].weights.T # matrix 
      else: 
        a = self.layers[L+1].biases.T # vektor (matrix in np)
      b = sigmoid_der(self.lincomb[L+1][:, self.currentbatch]) # vektor
      c = self._endgrad(L+1) # vektor
      return a @ (b * c) # vektor (scalar if bias)

  # gradient of w_ft
  def _basegrad(self):
    L = self.currentlayer
    if self.doweight:
      a = self.activations[L-1][self.currentrow, self.currentbatch] # scalar
      b = sigmoid_der(self.lincomb[L][:, self.currentbatch]) # vektor
    else:
      a = 1 # scalar
      b = sigmoid_der(self.lincomb[L][self.currentrow, self.currentbatch]) # scalar
    c = self._endgrad(L) # vektor (scalar if bias)
    return [a * b * c] # vektor

  # avg gradients of all columns fw
  def _avggrad(self):
    if self.doweight:
      cn = np.empty((0, self.layers[self.currentlayer].dto), float)
      appendaxis = 0
    else:
      cn = np.empty(0, float)
      appendaxis = None

    # go through every batch sample
    for self.currentbatch in range(BS):
      cn = np.append(cn, self._basegrad(), axis=appendaxis)
    # avg of batch samples in batch
    return np.sum(cn, axis=0) / BS # vektor

  def backward(self):
    print("**** BACK ****")
    cgradweights = np.empty((0, ), float)
    #cgradbias = np.array([])
    cgradbiases = np.empty((0, ), float)
    # go through all biases and weights from input layer
    for self.currentlayer in range(1, self.dimnum):
      for self.currentrow in range(self.layers[self.currentlayer].dfrom): # weights
        self.doweight = True
        cgradweights = np.append(cgradweights, self._avggrad())
      for self.currentrow in range(self.layers[self.currentlayer].dto):
        self.doweight = False
        cgradbiases = np.append(cgradbiases, self._avggrad())
    return cgradweights, cgradbiases

  # mean squarred error
  def cost(self, x, y, avg=False):
    c = squarederror(x, y)
    c = c.sum(axis=0, keepdims=True) 
    if avg:
      return np.sum(c) / c.shape[1]
    else: 
      return c

# bx, l0: a0, l0
m = model(784, 64, 16, 10)

for epoch in range(EPOCHS):
  # batch
  for i in range(0, imgnum, BS):
    bx, by = images[i:i+BS], labels[i:i+BS]
    bx, by = bx.T, by.T

    m.forward(bx, by)
    print("mean squared error", m.cost(m.activations[-1], by, 1))

    namblaw, namblab = m.backward()
    print("cn gradient w", namblaw, namblaw.shape)
    print("cn gradient b", namblab, namblab.shape)

    exit()
