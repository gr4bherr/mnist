#!/usr/bin/env python3
import numpy as np

# **** TRAIN ****
with open('train.npy', 'rb') as f:
  xtrain = np.load(f)
  ytrain = np.load(f)

imgnum, pxlnum = xtrain.shape

#np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

EPOCHS = 10
BS = 3
LR = 1e-3

class layer:
  def __init__(self, prevdim, dim):
    self.weights = np.random.rand(dim, prevdim) - 0.5
    self.biases = np.random.rand(dim, 1) - 0.5
  
  def z(self, x): # todo: better name
    return self.weights @ x + self.biases


class model:
  def __init__(self, *inp):
    self.layercnt = len(inp) // 2 + 1
    ls, af = inp[:self.layercnt], inp[self.layercnt:]
    self.layers, self.actfun = [None], [None] # first layers None
    for i in range(1, self.layercnt):
      self.layers.append(layer(ls[i-1], ls[i]))
      self.actfun.append(af[i-1])

  def _activationfunctions(self, i, x, forward=True):
    name = self.actfun[i]
    if forward:
      if name == "relu":
        return np.maximum(x, 0)
      elif name == "softmax":
        return np.exp(x) / sum(np.exp(x))
    # derivatives of activation functions
    else:
      if name == "relu":
        return x > 0
  def _squarrederror(self, x, y):
    return (x - y) ** 2
  def _squarrederror_der(self, x, y):
    return 2 * (x - y)

  def forward(self, x):
    self.z = [None] # z0 = None
    self.activations = [x] # a0 = x
    for i in range(1, self.layercnt):
      self.z.append(self.layers[i].z(self.activations[i-1]))
      self.activations.append(self._activationfunctions(i, self.z[i]))

  def backward(self, y):
    self.dz = []
    for i in range(self.layercnt - 1, 0, -1):
      if i == self.layercnt - 1:
        self.dz.append(self._squarrederror_der(self.activations[-1], y))
      else:
        self.dz.append(self.layers[i+1].weights.T @ self.dz[-1] * self._activationfunctions(i, self.z[i], False))
      self.layers[i].dweights = self.dz[-1] @ self.activations[i-1].T / BS
      self.layers[i].dbiases = np.sum(self.dz[-1], axis=1, keepdims=True) / BS

  def gradient(self):
    for i in range(1, self.layercnt):
      self.layers[i].weights = self.layers[i].weights - LR * self.layers[i].dweights
      self.layers[i].biases = self.layers[i].biases - LR * self.layers[i].dbiases 

  def accuracy(self, y):
    output = np.argmax(self.activations[-1], 0) 
    labels = np.argmax(y, 0)
    return np.sum(output == labels) / output.size

  # mean squared error loss function
  def loss(self, y):
    c = self._squarrederror(self.activations[-1], y)
    c = c.sum(axis=0, keepdims=True) 
    print("mean square error:", np.sum(c) / c.shape[1])


# ** RUN **
m = model(784, 16, 16, 10, "relu", "relu", "softmax")

for epoch in range(EPOCHS):
  # batch
  for i in range(0, imgnum, BS):
    #print(f"epoch: {epoch} batch: {i}")
    bx, by = xtrain[i:i+BS], ytrain[i:i+BS]
    bx, by = bx.T, by.T
    m.forward(bx)
    m.backward(by)
    m.gradient()
  m.loss(by)

# **** TEST ****
with open('test.npy', 'rb') as f:
  xtest = np.load(f)
  ytest = np.load(f)

acc = 0
for i in range(xtest.shape[0]):
  tx, ty = xtrain[i:i+1], ytrain[i:i+1]
  tx, ty = tx.T, ty.T
  m.forward(tx)
  acc += m.accuracy(ty)
print("accuracy on test:", acc / i)