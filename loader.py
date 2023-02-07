#!/usr/bin/env python3
import numpy as np

# load binary data and convert to numpy array

onehot = True

def loadData(name):
  tmp = []
  with open(f"mnist_data/MNIST/raw/{name}-ubyte", "rb") as f:
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
      if onehot:
        for x in arr[8:]:
          onehot = [0] * 10
          onehot[x] = 1
          tmp.append(onehot)
      else:
        tmp.append(x)
    return np.array(tmp)

with open('train.npy', 'wb') as f:
  np.save(f, loadData("train-images-idx3"))
  np.save(f, loadData("train-labels-idx1"))

with open('test.npy', 'wb') as f:
  np.save(f, loadData("t10k-images-idx3"))
  np.save(f, loadData("t10k-labels-idx1"))
