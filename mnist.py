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
#print(imgnum, pxlnum) 
#print()
#displayImage(0)

# 10 classes, 60 000 samples

#print("-------------------------------")
# softmax: exp -> noramlize
# categorical croos entropy

#print(images, labels)
print(images.shape, labels.shape)



# ACTIVATION FUNCTIONS
def randmatrix(r, c=0):
  # random values [-0.5, 0.5)
  return np.random.rand(r, c) - 0.5

#def layer(w, a, b):
  ##print(w.shape, a.shape, b.shape)
  #return w @ a - b

# def sigmoid(x):
  #return 1.0 / (1.0 + np.exp(x))

# def sigmoid_der(x):
#   return sigmoid(x) - (1 - simgoid(x))

# todo: undefinded in x = 0
def relu(a):
  # element wise maximum
  return np.maximum(0, a)
  #return np.array([0 if x < 0 else x for x in a])

def der_relu(a):
  return np.array([0 if x < 0 else 1 for x in a])

# logits -> prob
def softmax(a):
  #x = a - np.max(a, axis=1, keepdims=True) # numerical stability
  exp = np.exp(a)
  return exp / np.sum(exp, axis=0, keepdims=True)
  #x = a - np.max(a) # numerical stability (just to be sure)
  #exp = np.exp(x)
  #return exp / np.sum(exp)

# partial derivative of pi wrt xj
# &pi / &xj (& = partial)
# pi * (1 - pj) : if i=j
# -pj * pi      : if i!=j
def der_softmax(a):
  ones = np.ones(a.shape[0])
  return a @ (ones - a)

# def cost(x, l):
#   # cost per image in batch
#   return ((x - l) ** 2).sum(axis=0, keepdims=True)

EPOCHS = 1
BS = 25
LR = 1e-3

# init weights & biases
# num of weights: 784 * 64 + 64 * 10 = 50 816
# num of biases:        64 +      10  = 858


# gradient perpendicular to contour lines

# negative gradient of cost function = avg changes we want to make (to weights & biases) over all images




#def nambla

# a0 = bx    input layer
# w1 ?
# w1
# a1 --
# w2
# b2
# a2 -- 
# w3 
# b3
# a3 --      output layer



def sigmoid(z):
  return 1.0 / (1.0 + np.exp(z))

def sigmoid_der(z):
  return sigmoid(z) - (1 - sigmoid(z))


class layer:
  def __init__(self, f, t):
    self.weights = np.random.rand(t, f) - 0.5
    self.biases = np.random.rand(t, 1) - 0.5
    self.f = f # dim from
    self.t = t # dim to
  
  def z(self, x): # todo: better name
    return self.weights @ x - self.biases
  

# todo: add biases
# todo: editable only init and forward, rest automatic, inherit
class model:
  def __init__(self, *dims):
    #self.l = []
    #for i in range(len(dims) - 1):
      #self.l.append(layer(dims[i], dims[i+1]))
    self.l0 = None
    self.l1 = layer(784, 64)
    self.l2 = layer(64, 16)
    self.l3 = layer(16, 10)
    self.l = [self.l0, self.l1, self.l2, self.l3]
    #print(self.l0, self.l1, self.l2)
  
  def forward(self, x, y):
    self.y = y
    z0 = None
    a0 = x
    z1 = self.l[1].z(a0)
    a1 = sigmoid(z1)
    z2 = self.l[2].z(a1)
    a2 = sigmoid(z2)
    z3 = self.l[3].z(a2)
    a3 = sigmoid(z3)
    self.a = [a0, a1, a2, a3]
    self.z = [z0, z1, z2, z3]
    print(a0.shape, a1.shape, a2.shape, a3.shape)
    return a3
  
  # *** BACK ***

  def _endgrad(self, L):
    lnum = len(self.l) - 1
    print(L, lnum)

    if L > lnum:
      return 2 * (self.a[L] - self.y[L])
    else:
      s = []
      for tt in range(self.l[L].t):
        s += self.l[L].weights[tt] * sigmoid_der(self.z[L][tt, self.n]) * self._endgrad(L-1)

    exit()
    return l

  # gradient of w_ft
  def _basegrad(self):
    L = 1 # L
    #print(self.a[L-1][self.ff, self.n])
    #print(sigmoid_der(self.z[L][self.tt, self.n].shape))
    print(self._endgrad(L))
    return 50

  # avg gradients of all columns fw
  def _avggrad(self):
    cn = []
    for self.n in range(BS):
      #cn.append(self._basegrad(self.fw[:, n]))
      cn.append(self._basegrad())
    return sum(cn) / BS
    #return np.sum(grad(), axis=0) 


  def backward(self, fw):
    self.fw = fw
    print("**** BACK ****")
    cgrad = [] # row: 784 * 64 + 64; col: BS
    for self.ff in range(self.l[1].f): # weights
      cgrad.append(self._avggrad())
    # for self.tt in range(self.l[1].t):  # biases
    return np.array(cgrad)





  def cost(self, x, y, avg=False):
    c = ((x - y) ** 2).sum(axis=0, keepdims=True) 
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
    print("''''''", np.array(bx).shape)

    fw = m.forward(bx, by)
    print(m.cost(fw, by, 1))

    print("info:")
    print(m.l[1].weights.shape)
    print(m.l[1].biases.shape)
    #print(fw, fw.shape)

    g = m.backward(fw)
    print("cn gradient", g, g.shape)
    exit()








    # todo: this is just for one image at a time
    #for j in range(BATCHSIZE):
    #sample = i+j
    #tx, ty = images[sample:sample + 1], labels[sample:sample + 1]
    #tx, ty = tx.T, ty.T

    ## FORWARD PROP
    ## layer one
    #z1 = layer(w1, bx, b1)
    #print("z1", z1, z1.shape)
    #a1 = relu(z1)
    #print("a1", a1, a1.shape)
    ## layer two
    #z2 = layer(w2, a1, b2)
    #print("l2", z2, z2.shape)
    #a2 = softmax(z2)
    #print("a2", a2, a2.shape)
    #print("check", np.sum(a2, axis=0)) # sum of col
    #print("by", by, by.shape)
    #c = cost(a2, by) # c0, c1, ..., c24 (one for each image)
    #print("c", c, c.shape)
    #print("avg c", c.shape[1] / np.sum(c))
    ## BACK PROP

    #z1 = layer(w1, bx, b1)
    #a1 = sigmoid(z1)
    #z2 = layer(w2, a1, b2)
    #a2 = sigmoid(z2)
    #z3 = layer(w3, a2, b3)
    #a3 = sigmoid(z3)
    #print(a3, a3.shape)
    #print("by", by, by.shape)
    #c = cost(a3, by) # c0, c1, ..., c24 (one for each image)
    #print("c", c, c.shape)
    #print(np.sum(c))
    #print("avg c", np.sum(c) / c.shape[1])
    











    exit()







#def relu(inputs):
#  return np.maximum(0, inputs)
#
#def softmax(inputs):
#    expon = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
#    return expon / np.sum(expon, axis=1, keepdims=True)
#
#def d_relu(x):
#  return 1. if x > 0 else 0. # todo: remove .
#
#def layer(x, w, b):
#  print(x.shape, w.shape, b.shape)
#  return x @ w + b


## categorical cross entropy loss
#def loss(o, l):
  ## todo: clip
  ## low confidece -> higher loss
  #confidence = np.sum(o * l, axis=1)
  #neglog = -np.log(confidence)
  #print(confidence, confidence.shape)
  #print(neglog, neglog.shape)
  #print("adammmmmm")
  #print(np.sum(neglog))
  #return np.mean(neglog) # avg loss per batch

#def accuracy(o, l):
  #prediction = np.argmax(o, axis=1)
  #goal = np.argmax(l, axis=1)
  #print(prediction)
  #print(goal)
  #return np.mean(prediction == goal)
#print("=============")

#print("loss:", loss(a2, labels))

#print("accuracy", accuracy(a2, labels))

#print("myloss:")

#haha = -np.sum(labels * np.log(a2))

#print(haha)

#print(a2)