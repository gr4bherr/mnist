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
    self.dfrom = f # dim from
    self.dto = t # dim to
  
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
    self.dimnum = len(self.l)
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
    # number of layers

    # go through all nodes in next layer
    #for nodecnt in range(self.l[L].dto):
    if L == self.dimnum - 1: # last layer

      # loss function derivative
      return 2 * (self.a[L][:, self.n] - self.y[:, self.n]) # todo: always the same
    else:
      #print(L, "/", self.l[L].dto, end=" ")


      af = self.l[L+1].weights.T
      bf = sigmoid_der(self.z[L+1][:, self.n]) 
      cf = self._endgrad(L+1)
      #print(af.shape, bf.shape, cf.shape)
      #tmp.append(self.l[L+1].weights.T * sigmoid_der(self.z[L+1][:, self.n]) * self._endgrad(L+1))
      #print(af @ (bf * cf))
      #print((af@(bf*cf)).shape)
      return af @ (bf * cf)


  # gradient of w_ft
  def _basegrad(self, doweight):
    L = self.ll
    #if doweight:
      #print(self.a[self.ll-1][self.ff, self.n])
    #else:
      #print(1)

    sa = self.a[L-1][self.ff, self.n] 
    sb = sigmoid_der(self.z[L][:, self.n]) 
    sc = self._endgrad(L)
    #print(sa.shape, sb.shape, sc.shape)
    #print(sa)
    #return self.a[L-1][self.ff, self.n] * sigmoid_der(self.z[L][self.tt, self.n]) * self._endgrad(L)
    #print(sa * sb * sc)
    #print((sa * sb * sc).shape)
    sfuck = sa * sb * sc
    return [sa * sb * sc]

  # avg gradients of all columns fw
  def _avggrad(self, doweight=True):
    #cn = np.array([])
    cn = np.empty((0, self.l[self.ll].dto), float)
    #cn = []
    # go through every batch sample
    for self.n in range(BS):
      #cn.append(self._basegrad(self.fw[:, n]))
      cn = np.append(cn, self._basegrad(doweight), axis=0)
    #print("***DONE***")
    fuckme = np.sum(cn, axis=0) / BS
    #print(sum(fuckme))
    print(fuckme.shape)
    return fuckme
    #return np.sum(grad(), axis=0) 


  def backward(self, fw):
    self.fw = fw
    print("**** BACK ****")
    cgrad = np.array([]) # row: 784 * 64 + 64; col: BS
    # go through all biases and weights from input layer
    for self.ll in range(1, self.dimnum):
      print("ll", self.ll) # 1 - 3
      #for self.tt in range(self.l[self.ll].dto):  # biases
        #print(self.l[self.ll].dto)
        #print("tt", self.tt) # 0 - 63
      for self.ff in range(self.l[self.ll].dfrom): # weights
        #print(self.l[self.ll].dfrom)
        print("ff", self.ff, end=" ") # 0 - 784
        cgrad = np.append(cgrad, self._avggrad())
    print(cgrad.shape)
    exit()
        #cgrad.append(self._avggrad(False)) # todo: add biases
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













    exit()


