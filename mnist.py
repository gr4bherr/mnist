#!/usr/bin/env python3
import numpy as np

def displayImage(num):
  print(labels[num])
  for i in range(num, len(images)):
    print(f"{images[i][num]:.2f}", end=" ")
    if (i+1) % 28 == 0:
      print()

def loadData(name):
  tmp = []
  with open(f"dataset/{name}-ubyte", "rb") as f:
    arr = f.read()
    #2agicNum = int.from_bytes(arr[0:4], "big") # 2049: labels, 2051: images
    #itemNum = int.from_bytes(arr[4:8], "big")
    # images
    if int.from_bytes(arr[0:4], "big") == 2051: # 2051: image, 2049: labels
      # num of rows * num of columns
      pixelNum = int.from_bytes(arr[8:12], "big") * int.from_bytes(arr[12:16], "big")
      for i in range(16, len(arr), pixelNum):
        # pixel value as pct of 255
        tmp.append([arr[j]/255 for j in range(i, i + pixelNum)])
    # labels
    else: 
      # one hot encoding
      for x in arr[8:]:
        onehot = [0] * 10
        onehot[x] = 1
        tmp.append(onehot)
        #tmp.append(x)
    return np.array(tmp)

images = loadData("train-images-idx3")
labels = loadData("train-labels-idx1")

pxlNum, imgNum = images.shape # todo: rename to batch size or X (Xtrain, Xtest)
print(pxlNum, imgNum) 
print()
#displayImage(0)

print(images.shape)
print(labels.shape)

# 10 classes, 60 000 samples

print("-------------------------------")
# softmax: exp -> noramlize
# categorical croos entropy


# ACTIVATION FUNCTIONS
def relu(inputs):
  return np.maximum(0, inputs)

def softmax(inputs):
    expon = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    return expon / np.sum(expon, axis=1, keepdims=True)

# OTHER
def layer(x, w, b):
  return np.dot(x, w) + b

# categorical cross entropy loss
def loss(o, l):
  # todo: clip
  confidence = np.sum(o * l, axis=1)
  neglog = -np.log(confidence)
  return np.mean(neglog) 

def accuracy(o, l):
  prediction = np.argmax(o, axis=1)
  goal = np.argmax(l, axis=1)
  return np.mean(prediction == goal)

# init weights & biases
w1 = 0.1 * np.random.randn(784, 16)
b1 = np.zeros((1, 16))
w2 = 0.1 * np.random.randn(16, 10)
b2 = np.zeros((1, 10))

# FORWARD PROP
# input layer -> hidden layer
l1 = layer(images, w1, b1)
a1 = relu(l1)
# hiden layer -> output layer
l2 = layer(a1, w2, b2)
a2 = softmax(l2)

print("loss:", loss(a2, labels))

print("accuracy", accuracy(a2, labels))







