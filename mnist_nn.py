import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from mnist_nn_dataset import MnistNNTrain, MnistNNTest

class MnistNNLoad():
    def __init__(self, batchsize, worker):
        self.batchsize = batchsize
        self.worker = worker

    # Design a mini-batch gradient descent
    def load_data(self):
        batchsize = self.batchsize
        worker = self.worker
        train_dataset = MnistNNTrain()
        train_loader = DataLoader(dataset = train_dataset,
                    batch_size = batchsize,
                    shuffle = True,
                    num_workers = worker)
        test_dataset = MnistNNTest()
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

class MnistParameter():
    def __init__(self, 
                learning_rate, 
                momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    def mnist_function(self):
        learning_rate = self.learning_rate
        momentum = self.momentum
        model = MnistNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(),
                    lr = learning_rate, 
                    momentum = momentum)
        return model, criterion, optimizer

class RunMnistNN():
    def __init__(self, model, 
                criterion, optimizer, 
                train_loader, test_loader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train_mnist_nn(self):
        model = self.model
        model.train()
        criterion = self.criterion
        optimizer = self.optimizer
        train_loader = self.train_loader
        # train data in certain epoch
        for i, data in enumerate(train_loader):
            train_input, train_label = data
            train_input = np.array(train_input)
            train_label = np.array(train_label)
            # wrap them in Variable
            train_input = Variable(torch.Tensor(train_input), requires_grad = False)
            train_label = Variable(torch.LongTensor(train_label), requires_grad = False)
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(train_input)
            # Compute the train_loss with Cross-Entropy loss
            train_loss = criterion(y_pred, train_label)
            # Clear gradients of all optimized class
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        return float(train_loss)
    
    def test_mnist_nn(self):
        model = self.model
        model.eval()
        criterion = self.criterion
        optimizer = self.optimizer
        test_loader = self.test_loader
        for i, data in enumerate(test_loader):
            test_input, test_label = data
            test_input = np.array(test_input)
            test_label = np.array(test_label)
            test_input = Variable(torch.Tensor(test_input), requires_grad = False)
            test_label = Variable(torch.LongTensor(test_label), requires_grad = False)
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(test_input)
            test_loss = criterion(y_pred, test_label)
        return float(test_loss)