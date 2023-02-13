import torch
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

from sklearn.datasets import make_blobs
from torchvision.datasets import MNIST

    
class MyData():
  
  def make_loaders(self):
    self.TrainLoader = DataLoader(self.TrainData, batch_size=self.batch_size, shuffle=True)
    self.TestLoader = DataLoader(self.TestData, batch_size=self.batch_size, shuffle=False)
    self.SingleLoader = DataLoader(self.TrainData, batch_size=1, shuffle=False)
    self._SingleIterLoader = iter(self.SingleLoader)
    self.TrainOneBatch = DataLoader(self.TrainData, batch_size=len(self.TrainData), shuffle=False)
  
  def Next(self):
    try:
      return next(self._SingleIterLoader)
    except StopIteration:
      self._SingleIterLoader = iter(self.SingleLoader)
      return next(self._SingleIterLoader)


class ToyData(MyData):
  
  def __init__(self, n_features, classes, train_size=500, class_samples = 5000, batch_size = None, sphere = True, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    assert train_size < class_samples * classes
    self.n_features = n_features
    self.classes = classes
    self.class_samples = class_samples
    self.tot_size = class_samples * classes
    self.train_size = train_size
    self.test_size = self.tot_size - train_size
    if batch_size is None: batch_size = train_size
    self.batch_size = batch_size
    self.sphere = sphere
    self.device = device
    features, labels = make_blobs(n_features=n_features, n_samples=class_samples, centers=classes, cluster_std=.1, random_state=0)
    if sphere:
      for f in features:
        f /= np.linalg.norm(f)
    self.features = torch.Tensor(features).to(device)
    self.labels = torch.from_numpy(labels).to(device)
    self.TrainData = TensorDataset(self.features[:train_size], self.labels[:train_size])
    self.TestData = TensorDataset(self.features[train_size:], self.labels[train_size:])
    self.make_loaders()



"""
NB For standard MNIST the dimensions in the dataloder are (batch_size, 1, 28, 28), here just (batch_size, 784).
"""

class FastMNIST(MNIST):

    def __init__(self, *args, binary = False, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), **kwargs):
        super().__init__(*args, **kwargs)
        # Scale data to [0,1]
        self.data = self.data.float().div(255)
        self.data = self.data.sub_(0.1307).div_(0.3081)
        self.data = self.data.view(self.data.shape[0], -1)
        if binary:
          self.targets = torch.div(self.targets, 5, rounding_mode='floor')
        self.data, self.targets = self.data.to(device), self.targets.to(device)
        

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

class MNISTData(MyData):

  def __init__(self, batch_size = 128, binary = False, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    self.classes = 2 if binary else 10 #number of categories for classification
    self.n_features = 28*28 #dimension of the input space
    self.binary = binary
    self.batch_size = batch_size
    self.device = device
    self.TrainData = FastMNIST('./files/', train=True, binary=binary, device=device, download=True)
    self.TestData = FastMNIST('./files/', train=False, binary=binary, device=device, download=True)
    self.make_loaders()
    self.train_size = len(self.TrainData)
    self.test_size = len(self.TestData)
    self.tot_size = self.train_size + self.test_size



