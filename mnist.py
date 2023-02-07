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

def layer(w, a, b):
  print(w.shape, a.shape, b.shape)
  return w @ a - b

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

def cost(x, l):
  # small if correct (confident)
  return ((x - l) ** 2).sum(axis=1)


# todo: rename image, labels -> x, y
EPOCHS = 1
BATCHSIZE = 25
LR = 1e-3


# init weights & biases
# num of weights: 784 * 64 + 64 * 10 = 50 816
# num of biases:        64 +      10  = 858
w1 = randmatrix(64, 784)
b1 = randmatrix(64, 1)
w2 = randmatrix(10, 64)
b2 = randmatrix(10, 1)


#print(a2)
#print("cost", cost(a2, labels))
#print("avg cost", cost(a2, labels).sum() / imgnum)


#print()
#print(a2[0] , labels[0])
#print("changes we want in output", - (a2[0] - labels[0]))

# gradient perpendicular to contour lines

# negative gradient of cost function = avg changes we want to make (to weights & biases) over all images

for epoch in range(EPOCHS):
  # batch
  for i in range(0, imgnum, BATCHSIZE):
    #bx, by = images[i:i+BATCHSIZE], labels[i:i+BATCHSIZE]
    #bx, by = bx.T, by.T
    for j in range(BATCHSIZE):
      # INIT
      sample = i+j
      print("sample:", sample)
      tx, ty = images[sample:sample + 1], labels[sample:sample + 1]
      print(tx.shape, ty.shape)
      tx, ty = tx.T, ty.T
      print(tx.shape, ty.shape)
      # FORWARD PROP
      # layer one
      l1 = layer(w1, tx, b1)
      print("l1", l1, l1.shape)
      a1 = relu(l1)
      print("a1", a1, a1.shape)
      # layer two
      l2 = layer(w2, a1, b2)
      print("l2", l2, l2.shape)
      a2 = softmax(l2)
      print("a2", a2, a2.shape)
      print("check", np.sum(a2, axis=0))
      print("by", by, by.shape)
      # BACK PROP
      exit()











    exit()






