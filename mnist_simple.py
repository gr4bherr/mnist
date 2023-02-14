#!/usr/bin/env python3
import numpy as np

# **** TRAIN ****
with open('train.npy', 'rb') as f:
  xtrain = np.load(f)
  ytrain = np.load(f)

m, pxlnum = xtrain.shape

#np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

EPOCHS = 10
BS = 3
LR = 1e-3

def relu(x):
  return np.maximum(x, 0)
def relu_der(x):
  return x > 0

def softmax(x):
  return np.exp(x) / sum(np.exp(x))

class layer:
  def __init__(self, prevdim, dim):
    self.weights = np.random.rand(dim, prevdim) - 0.5
    self.biases = np.random.rand(dim, 1) - 0.5
  
  def z(self, x): # todo: better name
    return self.weights @ x + self.biases


class model:
  def __init__(self):
    l0 = None # input layer (784 neurons)
    l1 = layer(784, 10) # 1st hidden layer (10 neurons)
    l2 = layer(10, 10) # output layer (10 neurons)
    self.layers = [l0, l1, l2]
    self.lnum = len(self.layers)


  def forward(self, x):
    z0 = None
    a0 = x
    z1 = self.layers[1].z(a0)
    a1 = relu(z1)
    z2 = self.layers[2].z(a1)
    a2 = softmax(z2) # yhat
    self.linkombs = [z0, z1, z2]
    self.activations = [a0, a1, a2]
  
  def backward(self, y):
    dz2 = self.activations[2] - y # just minus the wanted value
    self.dw2 = 1 / BS * dz2 @ self.activations[1].T
    self.db2 = 1 / BS * np.sum(dz2, axis=1, keepdims=True)
    dz1 = self.layers[2].weights.T @ dz2 * relu_der(self.linkombs[1])
    self.dw1 = 1 / BS * dz1 @ self.activations[0].T
    self.db1 = 1/ BS * np.sum(dz1, axis=1, keepdims=True)

  def gradient(self):
    self.layers[1].weights = self.layers[1].weights - LR * self.dw1
    self.layers[1].biases = self.layers[1].biases - LR * self.db1
    self.layers[2].weights = self.layers[2].weights - LR * self.dw2
    self.layers[2].biases = self.layers[2].biases - LR * self.db2
  
  def accuracy(self, y):
    pred = np.argmax(self.activations[-1], 0)
    should = np.argmax(y, 0)
    return np.sum(pred == should) / pred.size

# squared error cost funciton  J = 1/m * sum((yhati - yi)^2)
# w = w - LR * loss_deriv, with tmp

mod = model()

# ** RUN **
for epoch in range(EPOCHS):
  # batch
  for i in range(0, m, BS):
    print(f"epoch: {epoch} batch: {i}")
    bx, by = xtrain[i:i+BS], ytrain[i:i+BS]

    bx, by = bx.T, by.T

    mod.forward(bx)

    mod.backward(by)

    #print("accuracy:", mod.accuracy(by))

    mod.gradient()

# **** TEST ****
with open('test.npy', 'rb') as f:
  xtest = np.load(f)
  ytest = np.load(f)

acc = 0
for i in range(xtest.shape[0]):
  tx, ty = xtrain[i:i+1], ytrain[i:i+1]
  tx, ty = tx.T, ty.T
  mod.forward(tx)
  acc += mod.accuracy(ty)
print("accuracy on test:", acc / i)