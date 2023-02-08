"""
pseudo code:

a0 (784) ---

w1 (784 * 64)
b1 (64)
a1 (64) ---- (sigmoid)

w2 (64 * 16)
b2 (16)
a2 (16) ---- (sigmoid)

w3 (16 * 10)
b3 (10)
a3 (10) ---- (sigmoid)
"""


# notes only on first of three
gradient = layergrad(784, 64) + layergrad(64, 16) + layergrad(16, 10)

# o -> 784
# p -> 64
def layergrad(f, t):
  c = []
  for j in range(t):
    for k in range(f): 
      c.append(w_grad(w_jk))
    c.append(b_grad(b_j))
  return c

def w_grad(x):


def b_grad(x):




def ():
  s = []
  for nn in range(n):
    s += ()
  return s / batchsize


# f : size of dim from
# t : size fo dim to
def layergradient(f, t):
  c = []
  for i in range(f * t):
    c.append(batchgradient)
  return c


