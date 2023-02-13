import torch
from math import sqrt
from scipy.special import xlogy
import os
import pickle

def KL_bin(q, p):
    return xlogy(q, q/p) + xlogy(1-q, (1-q)/(1-p))

def h(q, c, p):
    return KL_bin(q, p) - c

def hp(q, c, p):
    return (1-q)/(1-p) - q/p
    
def Newton_KL(q, c, p0, iter):
    p = p0
    for i in range(iter):
        p -= h(q, c, p) / hp(q, c, p)
    return p

def inv_KL(q, c, iter=5):
    b = q + sqrt(c/2)
    if b >= 1: return 1
    return Newton_KL(q, c, b, iter)

def KL_bin_torch(q, p):
    return torch.xlogy(q, q/p) + torch.xlogy(1-q, (1-q)/(1-p))

def h_torch(q, c, p):
    return KL_bin_torch(q, p) - c

def hp_torch(q, c, p):
    return (1-q)/(1-p) - q/p
    
def Newton_KL_torch(q, c, p0, iter):
    p = p0
    for i in range(iter):
        p = p - h_torch(q, c, p) / hp_torch(q, c, p)
    return p

def inv_KL_torch(q, c, iter=5, train=False):
    b = q + torch.sqrt(c/2)
    if b >= 1:
        if not train: return 1
        if train: #non-zero gradient for bound > 1
            return q + torch.sqrt(c/2)
    return Newton_KL_torch(q, c, b, iter)

__out_file__ = None
__term__ = True
__out_dir__ = './'

__OUT_DIR__ = lambda: __out_dir__
__OUT_FILE__ = lambda: __out_file__
__TERM__ = lambda: __term__

def Print(*args, **kwargs):
  term = __TERM__()
  out_file = __OUT_FILE__()
  if term: print(*args, **kwargs)
  if out_file is not None:
    out_file = os.path.join(__OUT_DIR__(), __OUT_FILE__())
    with open(out_file, 'a') as f:
      for arg in args:
        f.write(f'{arg}\n')
      f.flush()
      f.close()

def save_lists(*args, name=None, path=None):
  assert name is not None
  if path is None: path = __OUT_DIR__()
  path = os.path.join(path, name)
  with open(path, 'wb') as fp:
    pickle.dump(args, fp, protocol=pickle.HIGHEST_PROTOCOL)
