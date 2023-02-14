#!/usr/bin/env python3
import numpy as np

# math based on 3blue1brown

# **** TRAIN ****
with open('train.npy', 'rb') as f:
  images = np.load(f)
  labels = np.load(f)

imgnum, pxlnum = images.shape

EPOCHS = 1
BS = 25
LR = 1e-3

# negative gradient of cost function = avg changes we want to make (to weights & biases) over all images

def sigmoid(z):
  x = z - np.max(z) # numercial stabilitiy?
  return 1 / (1 + np.exp(-x))

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
      if self.weightorbias:
        a = self.layers[L+1].weights.T # matrix 
      else: 
        a = self.layers[L+1].biases.T # vektor (matrix in np)
      b = sigmoid_der(self.lincomb[L+1][:, self.currentbatch]) # vektor
      c = self._endgrad(L+1) # vektor
      return a @ (b * c) # vektor (scalar if bias)

  # gradient of w_ft
  def _basegrad(self):
    L = self.currentlayer
    if self.weightorbias:
      a = self.activations[L-1][self.currentrow, self.currentbatch] # scalar
      b = sigmoid_der(self.lincomb[L][:, self.currentbatch]) # vektor
    else:
      a = 1 # scalar
      b = sigmoid_der(self.lincomb[L][self.currentrow, self.currentbatch]) # scalar
    c = self._endgrad(L) # vektor (scalar if bias)
    return [a * b * c] # vektor

  # avg gradients of all columns fw
  def _avggrad(self):
    if self.weightorbias:
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
    cgradweights = np.empty((0, ), float)
    cgradbiases = np.empty((0, ), float)
    # go through all biases and weights from input layer
    for self.currentlayer in range(1, self.dimnum):
      # weights
      for self.currentrow in range(self.layers[self.currentlayer].dfrom):
        self.weightorbias = True
        cgradweights = np.append(cgradweights, self._avggrad())
      # biases
      for self.currentrow in range(self.layers[self.currentlayer].dto):
        self.weightorbias = False
        cgradbiases = np.append(cgradbiases, self._avggrad())
    return cgradweights, cgradbiases

  
  def optimizer(self):
    cunt = 0
    funt = 0
    for i in range(1, self.dimnum):
      for r in range(self.layers[i].dto): # for bias  (r, 0)
        # biases
        self.layers[i].biases[r][0] -= LR * namblab[funt]
        funt += 1
        for c in range(self.layers[i].dfrom):
          # weights
          self.layers[i].weights[r][c] -= LR * namblaw[cunt]
          cunt += 1 

  # mean squarred error
  def loss(self):
    c = squarederror(self.activations[-1], self.y)
    c = c.sum(axis=0, keepdims=True) 
    print("mean square error:", np.sum(c) / c.shape[1])
        
  def accuracy(self):
    out = np.argmax(self.activations[-1], axis=0)
    test = np.argmax(self.y, axis=0)
    x = [1 if out[i] == test[i] else 0 for i in range(len(out))]
    print("accuracy:", sum(x) / len(test))



m = model(784, 64, 16, 10)

for epoch in range(EPOCHS):
  # batch
  for i in range(0, imgnum, BS):
    bx, by = images[i:i+BS], labels[i:i+BS]
    bx, by = bx.T, by.T

    m.forward(bx, by)

    m.loss()

    namblaw, namblab = m.backward()

    m.optimizer()

    m.accuracy()

    #if i == 100 * BS:
      #break


exit()

#with open('test.npy', 'rb') as f:
  #testimages = np.load(f)
  #testlabels = np.load(f)

#testimgnum, testpxlnum = testimages.shape

#tx, ty = testimages[0:0+BS], testlabels[0:0+BS]
#tx, ty = tx.T, ty.T
#m.forward(tx, ty)
#m.accuracy()
