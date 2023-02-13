import os
from time import strftime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import math
import numpy as np

from WSN.tools import inv_KL, inv_KL_torch, Print, save_lists, __OUT_DIR__
from WSN.activations import __act__
from WSN.loss import Expected01Bin, Expected01

__methods__ = ['std', 'invKL', 'quad', 'lbd']

"""
StochBlock is the module that represents a stochastic layer.

INPUT
mother: nn.Linear block
device: torch.device

MAIN ATTRIBUTES
in_features, out_features: as in nn.Linear
device
weight: mean of the stochastic weight matrix
weight_in: weights when the module is created, (mean for the prior)
weight_v: variance of the weights, equal to 1/in_features, it keeps fixed during training
act: activation function

MAIN METHODS
forward: x -> W(act(x)), for a random realization of W different for each element in batch
OutLaw: for a single input (or for {repeat} inputs) it returns {repeat} realizations of the propagation
BatchOutLaw: as OutLaw, but it works on a batch, all the elements of the batch use the same random W per repetition
Penalty: returns the KL term between prior and posterior
extra_repr: print details
GetMQ: return the mean vector and the covariance matrix of the output in the Gaussian approximation
"""
class StochBlock(nn.Module):

  def __init__(self, mother, device):
    super(StochBlock, self).__init__()
    assert isinstance(mother, nn.Linear)
    self.in_features = mother.in_features
    self.out_features = mother.out_features
    self.device = device
    self.weight = Parameter(mother.weight.detach().clone(), requires_grad=True)
    self.weight_in = self.weight.clone().detach()
    weight_v = 1/self.in_features
    self.weight_v = torch.ones_like(self.weight.detach(), requires_grad=False, device=self.device)*weight_v #note that for now it is not registered as a parameter
    self.dim = self.in_features*self.out_features
    self.l_nb = mother.l_nb
    self.depth = mother.depth
    self.act = mother.act

  def forward(self, input):
    if self.act is not None:
      out = self.act['fct'](input)
    else:
      out = input
    A = torch.sqrt(F.linear(out**2, self.weight_v))
    M = F.linear(out, self.weight)
    N = torch.randn(size = M.shape, device=self.device, requires_grad=False)
    out = A * N + M
    return out

  def OutLaw(self, input, repeat): #input: [1, in_features] or [repeat, in_features]
    with torch.no_grad():
      if input.shape == (self.in_features,):
        input = input.unsqueeze(0)
      assert input.shape == (1, self.in_features) or (repeat, self.in_features)
      if self.act is not None:
        out = self.act['fct'](input)
      else:
        out = input
      A = torch.sqrt(F.linear(out**2, self.weight_v))
      M = F.linear(out, self.weight)
      N = torch.randn(size = (repeat, self.out_features), device=self.device, requires_grad=False)
      out = A * N + M
      return out
      
  def BatchOutLaw(self, input, repeat): #input: [batch, in_features] or [repeat, batch, in_features]
    with torch.no_grad():
      batch = input.shape[-2]
      if len(input.shape) == 2: input.unsqueeze(0)
      assert input.shape == (1, batch, self.in_features) or (repeat, batch, self.in_features)
      if self.act is not None:
        out = self.act['fct'](input)
      else:
        out = input
      A = torch.sqrt(F.linear(out**2, self.weight_v))
      M = F.linear(out, self.weight)
      N = torch.randn(size = (repeat, 1, self.out_features), device=self.device, requires_grad=False)
      out = A * N + M
      return out

  def Penalty(self):
    out = torch.sum((self.weight-self.weight_in)**2/(2*self.weight_v))
    return out

  def extra_repr(self):
    act = '' if self.act is None else self.act['name']+' '
    return f'in_features: {self.in_features}, out_features: {self.out_features}, action: x -> W {act}x'

  def _get_Aold_from_Qold(self, Qold):
    assert self.l_nb == 1
    Aold = torch.sqrt(Qold)
    return Aold
  
  def GetMQ(self, Mold, Qold):
    if self.l_nb == 0: #Mold = x, Qold and act_fct are ignored
      M = F.linear(Mold, self.weight)
      Q = F.linear(Mold**2, self.weight_v)
    else:
      Aold = self._get_Aold_from_Qold(Qold)
      P = self.act['m'](Aold, Mold)
      PP = self.act['Epr'](Qold, Mold, self.l_nb-1)
      M = F.linear(P, self.weight)
      Q = torch.diag_embed(F.linear(PP, self.weight_v)) + self.weight*(PP-P**2).unsqueeze(-2) @ self.weight.t()
    return M, Q



"""
GhostNet is a module containing methods used by Net and StochNet.
"""
class GhostNet(nn.Module):

  def __init__(self):
    super(GhostNet, self).__init__()

  def add_layer(self, name, layer):
    if not hasattr(self, 'layer_names'): self.layer_names = []
    self.layer_names.append(name)
    setattr(self, name, layer)

  def get_layers(self):
    return [getattr(self, layer) for layer in self.layer_names]

  def first_layer(self):
    return self.get_layers()[0]

  def last_layer(self):
    return self.get_layers()[-1]

  def forward(self, x):
    for layer in self.get_layers():
      x = layer.forward(x)
    return x

  def model_forward(self, x):
    for layer in self.get_layers()[:-1]:
      x = self.act['fct'](layer.forward(x))
    x = self.last_layer().forward(x)
    return x

  def selfie(self):
    return {
      'in_features': self.in_features,
      'out_features': self.out_features,
      'width': self.width,
      'depth': self.depth,
      'act_fct': self.act_fct}


"""
Net is the module for a deterministic network, with the same structure of the StochNet of interest.
It can be easily customized, but if non linear block are added, a stochastic equivalent should be implemented.

INPUT
in_features: input dimension
out_features: output dimension
width: number of nodes in hidden layer
depth: depth of the network (the corresponing StochNet is implemented only for depth=1)
act_fct: activation function (only 'relu' and 'sin' implemented)
device: torch.device

MAIN ATTRIBUTES
in_features, out_features, act_fct, width, depth

MAIN METHODS
forward: model_forward from GhostNet
StochSon: makes a StochNet centered on the model
"""

class Net(GhostNet):


  def __init__(self, in_features, out_features, width = 1000, depth=1, act_fct = 'relu', device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    assert act_fct in __act__, f"{act_fct} is not a valid activation function."
    self.device = device
    self.act_fct = act_fct
    self.act = __act__[act_fct]
    self.in_features = in_features
    self.out_features = out_features
    self.width = width
    self.depth = depth
    super(Net, self).__init__()
    first_layer = nn.Linear(in_features=self.in_features, out_features=self.width, bias=False)
    first_layer.l_nb = 0
    first_layer.act = None
    first_layer.depth = self.depth
    self.add_layer('lin0', first_layer)
    for idx in range(1, self.depth):
      l_name = f'lin{idx}'
      layer = nn.Linear(in_features=self.width, out_features=self.width, bias=False)
      layer.l_nb = idx
      layer.act = self.act
      layer.depth = self.depth
      self.add_layer(l_name, layer)
    last_layer = nn.Linear(in_features=self.width,  out_features=self.out_features, bias=False)
    last_layer.l_nb = self.depth
    last_layer.act = self.act
    last_layer.depth = self.depth
    self.add_layer(f'lin{self.depth}', last_layer)
    self.to(self.device)
    self.forward = self.model_forward
  
  @staticmethod
  def mirror(selfie, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    return Net(**selfie, device=device)

  def StochSon(self, delta=.025, name='SNN', beta=False):
    return StochNet(mother=self, delta=delta, name=name, beta=beta)



"""
StochNet is the stochastic version of Net, created by Net.StochSon.
Each linear block is converted in a StochBlock module.
It is a subclass of GhostNet, the structure of the forward method comes from there.
Implemented only for depth=1 (single hidden layer), unless using beta version.

INPUTS
mother: the model to be made stochastic
delta: PAC parameter delta for the bound (default .025)
beta: if True uses beta version and allows multilayer Net as mother (default False)

MAIN ATTRIBUTES
width, depth, delta, in_features, out_features
dim: the dimension of the parameter space (sum of dimensions of weight matrices)
act_fct: activation function

MAIN METHODS
forward: stochastic realization of the ouput, each element in batch is propagated with an independent realization
Save: saves the model in {path}, if path is not given it is automatically generated with a timestamp
Load: loads a saved model from {path}
GetMQ: returns the mean vector and the covariance matrix of the output in the Gaussian approximation
Penalty: evaluates the penalty for the PAC Bayes upperbound, '(KL + log(2 * √train_size / delta) / train_size'
Train: trains via standard PAC-Bayesian methods
GaussTrain: trains via the Gaussian method
TrainingSchedule: runs Train or GaussTrain for several epochs with different learning rates
Test: tests the error in the standard way, for each input there is a different random realization of the network
GaussTest: tests the error assuming the exactness of the Gaussian approximation
PrintBound: PAC-Bayes bound
GaussBound: PAC-Bayes bound evaluated assuming the exactness of the Gaussian approximation
OutLaw: for a single input returns {repeat} outputs obtained via independent realizations of the network
"""

class StochNet(GhostNet, nn.Module):


  def __init__(self, mother, delta=.025, name='SNN', beta=False):
    super(StochNet, self).__init__()
    self.name = name
    self.mother_state = mother.state_dict()
    self.device = mother.device
    self.act_fct = mother.act_fct
    self.act = mother.act
    self.in_features = mother.in_features
    self.out_features = mother.out_features
    self.width = mother.width
    self.depth = mother.depth
    self.beta = beta
    if not beta: assert self.depth == 1, "Multilayer StochNet not implemented."
    else: Print('Warning: Creating a beta StochNet {name}, Gaussian training not available')
    self.dim = 0
    self.best = 1.
    layers = mother._modules
    self.layer_names = mother.layer_names
    for key in layers:
      SL = StochBlock(layers[key], device=self.device)
      setattr(self, key, SL)
      self.dim += SL.dim
    self.delta = delta
    self.Lambda = torch.zeros(1, device=self.device, requires_grad=True, dtype=torch.float32)
    self.to(self.device)

  @staticmethod
  def mirror(info, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), name='SNN'):
    M = Net.mirror(info['selfie'], device=device)
    M.load_state_dict(info['mother'])
    if 'beta' not in info.keys(): info['beta'] = False #for compatibility issues with previous version where beta was not defined
    SNN = M.StochSon(info['delta'], name=name, beta = info['beta'])
    SNN.Lambda = torch.tensor([info['Lambda']], device=device, requires_grad=True, dtype=torch.float32)
    return SNN

  def info(self):
    return {
      'selfie': self.selfie(),
      'mother': self.mother_state,
      'delta': self.delta,
      'Lambda': self.Lambda.item(),
      'beta': self.beta
    }

  def Save(self, name=None, path=None, timestamp = None):
    if name is None: name = self.name
    if path is None: path = __OUT_DIR__()
    if timestamp is None: timestamp = strftime("%Y%m%d-%H%M%S")
    if timestamp != '': name += '_' + timestamp
    dirpath = os.path.join(path, name)
    if os.path.exists(dirpath):
      try: assert os.path.isdir(dirpath)
      except AssertionError:
        Print(f'Could not save in {dirpath}')
    else: os.mkdir(dirpath)
    info_path = os.path.join(dirpath, 'info')
    dict_path = os.path.join(dirpath, 'state_dict')
    torch.save(self.state_dict(), dict_path)
    torch.save(self.info(), info_path)
    return dirpath

  @staticmethod
  def Load(path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), name='SNN'):
    info_path = os.path.join(path, 'info')
    dict_path = os.path.join(path, 'state_dict')
    info = torch.load(info_path, map_location=device)
    state_dict = torch.load(dict_path, map_location=device)
    SNN = StochNet.mirror(info, device=device, name=name)
    SNN.load_state_dict(state_dict)
    return SNN

  def GetMQ(self, x):
    M = x
    Q = None
    for layer in self.get_layers():
      M, Q = layer.GetMQ(M, Q)
    return M, Q

  def Penalty(self, train_size):
    out = sum(l.Penalty() for l in self._modules.values())
    out = out + math.log(2*train_size**.5/self.delta)
    out = out / train_size
    return out

  def TrainingSchedule(self, dataloader, EPOCH, LR, gauss=True, timestamp=None, save_track=True, **kwargs):
    assert type(EPOCH)==list and type(LR)==list
    assert len(EPOCH) == len(LR)
    schedule = zip(EPOCH, LR)
    if timestamp is None: timestamp = strftime("%Y%m%d-%H%M%S")
    Print('************* Starting training schedule *************')
    OUT = [[], [], [], []]
    for epoch, lr in schedule:
      if gauss: out = self.GaussTrain(dataloader, epoch, lr, timestamp=timestamp, **kwargs)
      else: out = self.Train(dataloader, epoch, lr, timestamp=timestamp, **kwargs)
      if type(out) is tuple: #track=True, track_lbd might be True or False
        for i, o in enumerate(out):
          OUT[i] += o
      elif type(out) is list: #track=False, track_lbd=True
        OUT[0].append(out)
    Print('************* Training schedule completed *************')
    if len(OUT[0]) > 0:
      if len(OUT[1]) == 0:
        return OUT[0]
      else:
        out = []
        for o in OUT:
          if len(o)>0: out.append(o)
        out = tuple(out)
        if save_track:
          name = 'Progr_' + self.name
          if timestamp != '': name += '_' + timestamp
          save_lists(*out, name=name)
        return out

  """
  GaussTrain settings:
  multi: if True uses Expected01 loss even for binary classification
  samples: number of realizations of the MC estimate in Expected01 loss
  track: if True returns the evolutions of loss, penalty and bound during training
  method: cf the experimental section in the paper, 'invKL'='G std', 'quad'='G quad', 'lbd'='G lbd'
  penalty: weight of the KL penalty
  pmin: for methods 'quad' and 'lbd' minimal values of output probabilities for multiclass classification
  lbd_in: for method 'lbd', initial value of the additional parameter lbd
  lr_lbd: for method 'lbd' learning rate in lbd training
  track_lbd: if true returns evolution of lbd during training
  no_save: if False and track is True, the best configuration found of the networks is automatically saved
  """
  def GaussTrain(self, dataloader, epoch, lr, multi=False, samples=10**4, track=False, method='invKL', penalty=1, pmin=None, lbd_in=None, lr_lbd=0.001, track_lbd=False, no_save=False, timestamp=None):
    assert not self.beta, "Gauss methods not available for beta StochNet"
    assert method in __methods__
    Print(f'Training started -- Epochs: {epoch} -- lr: {lr} -- method: {method}')
    if track:
      loss_list = []
      KL_list = []
      bound_list = []
      if not no_save:
        if timestamp is None: timestamp = strftime("%Y%m%d-%H%M%S")
    train_size = sum([len(data) for data, _ in dataloader])
    optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
    if method == 'lbd':
      if lbd_in is not None:
        assert lbd_in<1 and lbd_in>0
        self.Lambda = torch.tensor([math.atanh(2*lbd_in-1)], device=self.device, requires_grad=True, dtype=torch.float32)
      optimizer_lbd = torch.optim.SGD([self.Lambda], lr=lr_lbd, momentum=0.9)
      lbd_epoch = False
      epoch = 2*epoch
    if track_lbd: lbd_list = []
    for ep in range(epoch):
      if method != 'lbd':
        Print(f'Starting epoch {ep+1}')
        opt = optimizer
        ep_track = track
        track_lbd = False
        ep_track_lbd = False
      else:
        if lbd_epoch:
          Print(f'Starting epoch {ep//2+1} - lbd')
          opt = optimizer_lbd
          ep_track = track
          ep_track_lbd = track_lbd
        else:
          Print(f'Starting epoch {ep//2+1}')
          opt = optimizer
          ep_track = False
          ep_track_lbd = False
        lbd_epoch = not lbd_epoch
      running_loss = 0
      tot = 0
      if ep_track:
        KL_track = 0
        loss_track = 0
        bound_track = 0
      for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.to(self.device)
        batch_size = len(data)
        targets = targets.to(self.device)
        opt.zero_grad()
        KL = self.Penalty(train_size)
        if ep_track: KL_track += KL.item()*batch_size
        if multi or self.out_features>2:
          loss = Expected01(self, data, targets, samples=samples)
          if ep_track:
            with torch.no_grad():
              loss_track += loss.item()*batch_size
              bound_track += inv_KL(loss.item(), KL.item())*batch_size
        else:
          loss = Expected01Bin(self, data, targets)
          if ep_track:
            with torch.no_grad():
              loss_track += loss.item()*batch_size
              bound_track += inv_KL(loss.item(), KL.item())*batch_size
        if method == 'invKL':
          loss = inv_KL_torch(loss, penalty*KL, train=True)
        elif method == 'quad':
          loss = (torch.sqrt(loss + penalty*KL/2) + torch.sqrt(penalty*KL/2))**2
        elif method == 'std':
          loss = loss + torch.sqrt(penalty*KL/2)
        elif method == 'lbd':
          loss = 4*(loss + 2*penalty*KL/(1+torch.tanh(self.Lambda)))/(3-torch.tanh(self.Lambda))
        loss.backward()
        opt.step()
        tot += batch_size
        running_loss += loss.item()*batch_size
      if ep_track:
        loss_list.append(loss_track/tot)
        KL_list.append(KL_track/tot)
        bound_list.append(bound_track/tot)
        if not no_save:
          if bound_track/tot < self.best:
            self.best = bound_track/tot
            self.Save(name='Best_'+self.name, timestamp=timestamp)
      if ep_track_lbd: lbd_list.append((1+math.tanh(self.Lambda.item()))/2)
      if method != 'lbd': Print(f'Epoch {ep+1} completed -- Average loss {running_loss/tot:.5f}')
      elif not lbd_epoch: Print(f'Epoch {ep//2+1} completed -- Average loss {running_loss/tot:.5f}')
    Print('Training Completed')
    if track and not track_lbd: return loss_list, KL_list, bound_list
    if track and track_lbd: return loss_list, KL_list, bound_list, lbd_list
    if track_lbd: return lbd_list

  """
  Train settings:
  track: if True returns the evolutions of loss, penalty and bound during training
  method: cf the experimental section in the paper, 'std'='S std', 'quad'='S quad', 'lbd'='S lbd'
  penalty: weight of the KL penalty
  pmin: for methods 'quad' and 'lbd' minimal values of output probabilities for multiclass classification
  lbd_in: for method 'lbd', initial value of the additional parameter lbd
  lr_lbd: for method 'lbd' learning rate in lbd training
  track_lbd: if true returns evolution of lbd during training
  samples: used by Expected01 during track, does not affect the training, for faster training set track to False or samples to a smaller value
  no_save: if False and track is True, the best configuration found of the networks is automatically saved
  """

  def Train(self, dataloader, epoch, lr, track=False, method='std', pmin=None, penalty=1, lbd_in=0.5, lr_lbd=0.001, track_lbd=False, samples=10**4, no_save=False, timestamp=None):
    assert method in __methods__
    if track: assert not self.beta, 'No tracking available in training of StochNet'
    Print(f'Training started -- epochs: {epoch} -- lr: {lr} -- method: {method}')
    train_size = sum([len(data) for data, _ in dataloader])
    optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
    if method == 'lbd':
      if lbd_in is not None:
        assert lbd_in<1 and lbd_in>0
        self.Lambda = torch.tensor([math.atanh(2*lbd_in-1)], device=self.device, requires_grad=True, dtype=torch.float32)
      optimizer_lbd = torch.optim.SGD([self.Lambda], lr=lr_lbd, momentum=0.9)
      lbd_epoch = False
      epoch = 2*epoch
    CEL = nn.CrossEntropyLoss()
    NLLL = nn.NLLLoss()
    if track:
      loss_list = []
      KL_list = []
      bound_list = []
      if not no_save:
        if timestamp is None: timestamp = strftime("%Y%m%d-%H%M%S")
    if track_lbd: lbd_list = []
    for ep in range(epoch):
      if method != 'lbd':
        Print(f'Starting epoch {ep+1}')
        opt = optimizer
        ep_track = track
        track_lbd = False
        ep_track_lbd = False
      else:
        if lbd_epoch:
          Print(f'Starting epoch {ep//2+1} - lbd')
          opt = optimizer_lbd
          ep_track = track
          ep_track_lbd = track_lbd
        else:
          Print(f'Starting epoch {ep//2+1}')
          opt = optimizer
          ep_track = False
          ep_track_lbd = False
        lbd_epoch = not lbd_epoch
      running_loss = 0
      tot = 0
      if ep_track:
        KL_track = 0
        loss_track = 0
        bound_track = 0
      for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.to(self.device)
        batch_size = len(data)
        targets = targets.to(self.device)
        opt.zero_grad()
        KL = self.Penalty(train_size)
        if ep_track: KL_track += KL.item()*batch_size
        outputs = self.forward(data)
        if self.out_features == 2 and pmin is None:
          loss = CEL(outputs, targets)/math.log(2)
        else:
          if pmin is None: pmin = 10**-5
          probs = torch.clamp(torch.softmax(outputs, dim=-1), min=pmin)
          loss = NLLL(torch.log(probs), targets)/math.log(1/pmin)
        if ep_track:
          with torch.no_grad():
            if self.out_features == 2 :
              loss01 = Expected01Bin(self, data, targets).item()
            else: loss01 = Expected01(self, data, targets, samples=samples).item()
            loss_track += loss01*batch_size
            bound_track += inv_KL(loss01, KL.item())*batch_size
        if method == 'invKL':
          loss = inv_KL_torch(loss, penalty*KL, train=True)
        elif method == 'quad':
          loss = (torch.sqrt(loss + penalty*KL/2) + torch.sqrt(penalty*KL/2))**2
        elif method == 'std':
          loss = loss + torch.sqrt(penalty*KL/2)
        elif method == 'lbd':
          loss = 4*(loss + 2*penalty*KL/(1+torch.tanh(self.Lambda)))/(3-torch.tanh(self.Lambda))
        loss.backward()
        opt.step()
        tot += batch_size
        running_loss += loss.item()*batch_size
      if ep_track:
        loss_list.append(loss_track/tot)
        KL_list.append(KL_track/tot)
        bound_list.append(bound_track/tot)
        if not no_save:
          if bound_track/tot < self.best:
            self.best = bound_track/tot
            self.Save(name='Best_'+self.name, timestamp=timestamp)
      if ep_track_lbd: lbd_list.append((1+math.tanh(self.Lambda.item()))/2)
      if method != 'lbd': Print(f'Epoch {ep+1} completed -- Average loss {running_loss/tot:.5f}')
      elif not lbd_epoch: Print(f'Epoch {ep//2+1} completed -- Average loss {running_loss/tot:.5f}')
    Print('Training Completed')
    if track and not track_lbd: return loss_list, KL_list, bound_list
    if track and track_lbd: return loss_list, KL_list, bound_list, lbd_list
    if track_lbd: return lbd_list

  def GaussTest(self, dataloader, quiet=False):
    assert not self.beta, "Gauss methods not available for beta StochNet"
    tot_loss = 0.
    total = 0
    if self.out_features==2: Loss = lambda x, y: Expected01Bin(self, x, y)
    else: Loss = lambda x,y : Expected01(self, x, y, samples=10**5)
    with torch.no_grad():
      for idx, (data, target) in enumerate(dataloader):
        data = data.to(self.device)
        target = target.to(self.device)
        dim_b = target.size(0)
        tot_loss += Loss(data, target).item()*dim_b
        total += dim_b
    if not quiet: Print('Accuracy of the network: %0.3f %%' % (100 * (1-tot_loss/total)))
    return tot_loss/total

  def Test(self, dataloader, quiet=False):
    self.eval()
    correct = 0
    total = 0
    with torch.no_grad():
      for idx, (data, target) in enumerate(dataloader):
        data = data.to(self.device)
        target = target.to(self.device)
        outputs = self.forward(data)
        guess = torch.max(outputs, -1)[1]
        total += target.size(0)
        correct += (guess == target).sum().item()
    if not quiet: Print('Accuracy of the network: %0.3f %%' % (100 * correct / total))
    return correct/total

  def _out_law(self, x, repeat):
    for layer in self.get_layers():
      x = layer.OutLaw(x, repeat=repeat)
    return x.detach().cpu().numpy()

  def OutLaw(self, x, repeat, repeat_limit=None):
    #if the gpu is not large enough, use a small repeat_limit
    if repeat_limit is None: repeat_limit = repeat
    else: repeat_limit = min(repeat, repeat_limit)
    repeat_list = (repeat // repeat_limit) * [repeat_limit]
    reminder = repeat % repeat_limit
    if reminder > 0: repeat_list.append(reminder)
    out_list = []
    with torch.no_grad():
      out_list = []
      for rep in repeat_list:
        out_list.append(self._out_law(x, rep))
      out = np.concatenate(out_list)
    return out.T #output: [out_features, repeat]

  def _batch_out_law(self, x, repeat):
    for layer in self.get_layers():
      x = layer.BatchOutLaw(x, repeat=repeat)
    return x

  def PrintBound(self, dataloader, N_nets, repeat_limit=None, quiet=False, deltap=0.01):
    #if the gpu is not large enough, use a small repeat_limit
    #for each batch per each repetition a single realization of the stochastic network is used
    repeat = N_nets
    if repeat_limit is None: repeat_limit = repeat
    else: repeat_limit = min(repeat, repeat_limit)
    repeat_list = (repeat // repeat_limit) * [repeat_limit]
    reminder = repeat % repeat_limit
    if reminder > 0: repeat_list.append(reminder)
    correct = 0
    total = 0
    current_rep = 0
    with torch.no_grad():
      for rep in repeat_list:
        current_rep += rep
        for idx, (data, target) in enumerate(dataloader):
          outputs = self._batch_out_law(data, rep)
          guess = torch.max(outputs, -1)[1]
          total += rep*target.size(0)
          correct += (guess == target.unsqueeze(0)).sum().item()
          if not quiet: Print(f'Batch {idx+1} of {len(dataloader)} --- Rep {current_rep} of {repeat} --- Current average score {correct/total:.5e}')
      emp_loss = 1 - correct/total
      Print(f'Estimated empirical loss: {emp_loss}')
      emp_loss = inv_KL(emp_loss, math.log(2/deltap)/repeat)
      Print(f'Upper bound on the empirical loss: {emp_loss}')
      penalty = self.Penalty(train_size=total/repeat)
      Print(f'Penalty: {penalty.item()}')
      bound = inv_KL(emp_loss, penalty.item())
    Print(f'With P >= {1-self.delta-deltap} the true error is bounded by {bound}')
    return bound

  def TestError(self, dataloader, N_nets=1000, repeat_limit=None, quiet=True, std=True):
    #if the gpu is not large enough, use a small repeat_limit
    #for each batch per each repetition a single realization of the stochastic network is used
    repeat = N_nets
    if repeat_limit is None: repeat_limit = repeat
    else: repeat_limit = min(repeat, repeat_limit)
    repeat_list = (repeat // repeat_limit) * [repeat_limit]
    reminder = repeat % repeat_limit
    if reminder > 0: repeat_list.append(reminder)
    current_rep = 0
    SCORES = torch.tensor([], device=self.device)
    with torch.no_grad():
      for rep in repeat_list:
        current_rep += rep
        scores = torch.zeros(rep, device=self.device)
        total_size = 0
        for idx, (data, target) in enumerate(dataloader):
          batch_size = target.size(0)
          outputs = self._batch_out_law(data, rep)
          guess = torch.max(outputs, -1)[1]
          total_size += batch_size
          scores += (guess == target.unsqueeze(0)).sum(1)
          if not quiet: Print(f'Batch {idx+1} of {len(dataloader)} --- Rep {current_rep} of {repeat}')
        SCORES = torch.cat((SCORES, scores/total_size))
        Print(f'Repetitions: {current_rep} of {repeat} --- Estimated test loss: {1-SCORES.mean()}')
      error_mean = 1 - SCORES.mean().item()
      if std:
        error_std = SCORES.std().item()
        return error_mean, error_std
      else: return error_mean

  def GaussBound(self, dataloader, train_size=None, emp_loss=None):
    assert not self.beta, "Gauss methods not available for beta StochNet"
    with torch.no_grad():
      if train_size is None:
        train_size = sum([len(data) for data, _ in dataloader])
      if emp_loss is None: emp_loss = self.GaussTest(dataloader)
      penalty = self.Penalty(train_size).item()
      bound = inv_KL(emp_loss, penalty)
    Print(f'Penalty {penalty}')
    Print(f'Emp loss {emp_loss}')
    Print(f'If the Gaussian approximation is exact, with P >= {1-self.delta} the true error is bounded by {bound}')
    return bound



