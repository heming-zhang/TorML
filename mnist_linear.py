import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from mnist_dataset import MnistLinearTrain, MnistLinearTest

class MnistLinear():
    def __init__(self, batchsize, worker):
        self.batchsize = batchsize
        self.worker = worker

    # Design a mini-batch gradient descent
    def load_data(self):
        batch_size = self.batch_size
        num_worker = self.num_worker
        train_dataset = MnistLinearTrain()
        train_loader = DataLoader(dataset = train_dataset,
                    batch_size = batchsize,
                    shuffle = True,
                    num_workers = worker)
        test_dataset = MnistLinearTest()
        test_loader = DataLoader(dataset = test_dataset,
                    batch_size = batchsize,
                    shuffle = True,
                    num_workers = worker)
        return train_loader, test_loader

class MnistNN(nn.Module):
    def __init__(self):
        super(MnistNN, self).__init__()
        self.l1 = torch.nn.Linear(784, 560)
        self.l2 = torch.nn.Linear(560, 440)
        self.l3 = torch.nn.Linear(440, 210)
        self.l4 = torch.nn.Linear(210, 80)
        self.l5 = torch.nn.Linear(80, 10)
    
    # Use ReLU as activation function
    def forward(self, x):
        out1 = F.relu(self.l1(x))
        out2 = F.relu(self.l2(out1))
        out3 = F.relu(self.l3(out2))
        out4 = F.relu(self.l4(out3))
        y_pred = self.l5(out4)
        return y_pred

class RunMnistNN():
    def __init__(self, learning_rate, momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    def train_mnist_nn(self):
        learning_rate = self.learning_rate
        momentum = self.momentum
        model = MnistNN()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(),
                    lr = learning_rate, 
                    momentum = momentum)
