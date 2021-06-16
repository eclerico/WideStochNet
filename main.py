from network import Net, StochNet
from datasets import ToyData, MNISTData
import tools
from tools import Print

import torch
import numpy as np

import os, sys
from time import strftime

#set output dir
outpath = f'./OUTPUT_{strftime("%Y%m%d-%H%M%S")}'
tools.__out_dir__ = outpath
try: assert os.path.exists(outpath)
except AssertionError: os.mkdir(outpath)
assert os.path.isdir(outpath)

tools.__out_file__ = 'output' #name of the log file
tools.__term__ = True #print output on the terminal

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Print(f'Running on {device}')

#Training schedule
EPOCH = [100, 1000, 500]
LR = [0.01, 0.001, 0.0001]

#StochNet on ToyData
DATA = ToyData(n_features=3, classes=2)
classes = DATA.classes
n_features=DATA.n_features
TL = DATA.TrainLoader

width = 1200
NN = Net(out_features=classes, in_features=n_features, width=width, act_fct='sin') #network with sin activation
SNN = NN.StochSon(name = 'SN_Toy')
loss, KL, bound = SNN.TrainingSchedule(TL, EPOCH, LR, track=True, gauss=True, method='invKL') #training with Gaussian method 'G std'
SNN.Save()
SNN.GaussBound(TL)
SNN.PrintBound(TL, N_nets=10000, repeat_limit=100)

#StochNet on MNIST
DATA = MNISTData() #use MNISTData(binary=True) for binary MNIT
classes = DATA.classes
n_features = DATA.n_features
TL = DATA.TrainLoader

width = 1200
NN = Net(out_features=classes, in_features=n_features, width=width, act_fct='relu') #network with relu activation
SNN = NN.StochSon(name = 'SN_MNIST')

loss, KL, bound = SNN.TrainingSchedule(TL, EPOCH, LR, track=True, gauss=False, method='lbd') #training with standard method 'S lbd'
SNN.Save()
SNN.GaussBound(TL)
SNN.PrintBound(TL, N_nets=10000, repeat_limit=100)

