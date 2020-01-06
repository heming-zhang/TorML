import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from mnist_cnn_dataset import MnistCNNTrain, MnistCNNTest

class MnistCNNLoad():
    def __init__(self, batchsize, worker):
        self.batchsize = batchsize
        self.worker = worker

    # Design a mini-batch gradient descent
    def load_data(self):
        batchsize = self.batchsize
        worker = self.worker
        train_dataset = MnistCNNTrain()
        train_loader = DataLoader(dataset = train_dataset,
                    batch_size = batchsize,
                    shuffle = True,
                    num_workers = worker)
        test_dataset = MnistCNNTest()
        test_loader = DataLoader(dataset = test_dataset,
                    batch_size = batchsize,
                    shuffle = True,
                    num_workers = worker)
        return train_loader, test_loader

class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()

    
    # Use ReLU as activation function
    def forward(self, x):
        out1 = F.relu(self.l1(x))
        out2 = F.relu(self.l2(out1))
        out3 = F.relu(self.l3(out2))
        out4 = F.relu(self.l4(out3))
        y_pred = self.l5(out4)
        return y_pred

class MnistCNNParameter():
    def __init__(self, 
                learning_rate, 
                momentum,
                cuda):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.cuda = False
    
    def mnist_function(self):
        learning_rate = self.learning_rate
        momentum = self.momentum
        cuda = self.cuda
        if cuda:
            model = MnistNN().cuda()
        else:
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
        # Train data in certain epoch
        train_correct = 0
        for i, data in enumerate(train_loader):
            train_input, train_label = data
            train_input = np.array(train_input)
            train_label = np.array(train_label)
            # Wrap them in Variable
            train_input = Variable(torch.Tensor(train_input),
                        requires_grad = False)
            train_label = Variable(torch.LongTensor(train_label),
                        requires_grad = False)
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(train_input)
            # Compute the train_loss with Cross-Entropy loss
            train_loss = criterion(y_pred, train_label)
            # Clear gradients of all optimized class
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # Compute accuracy rate
            _, train_pred = torch.max(y_pred.data, 1)
            train_correct += (train_pred == train_label).sum()
        train_accuracy = float(train_correct) / len(train_loader.dataset)
        return float(train_loss), train_accuracy
    
    def test_mnist_nn(self):
        model = self.model
        model.eval()
        criterion = self.criterion
        optimizer = self.optimizer
        test_loader = self.test_loader
        # Test model accuracy after certain epoch
        test_correct = 0
        for i, data in enumerate(test_loader):
            test_input, test_label = data
            test_input = np.array(test_input)
            test_label = np.array(test_label)
            test_input = Variable(torch.Tensor(test_input), requires_grad = False)
            test_label = Variable(torch.LongTensor(test_label), requires_grad = False)
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(test_input)
            test_loss = criterion(y_pred, test_label)
            _, test_pred = torch.max(y_pred, 1)
            test_correct += (test_pred == test_label).sum()
        test_accuracy = float(test_correct) / len(test_loader.dataset) 
        return float(test_loss), test_accuracy

class MnistNNPlot():
    def __init__(self, train_accuracy_rate, 
                test_accuracy_rate,
                epoch_num):
        self.train_accuracy_rate = train_accuracy_rate
        self.test_accuracy_rate = test_accuracy_rate
        self.epoch_num = epoch_num

    def plot(self):
        train_accuracy_rate = self.train_accuracy_rate
        test_accuracy_rate = self.test_accuracy_rate
        epoch_num = self.epoch_num
        # Plot train and test accuracy rate
        x = range(1, epoch_num + 1)
        plt.plot(x, train_accuracy_rate, 
                    label = "Mnist-NN-Train-Accuracy-Rate")
        plt.plot(x, test_accuracy_rate, 
                    label = "Mnist-NN-Test-Accuracy-Rate")
        plt.xlabel("Epoch Time")
        plt.ylabel("Accuracy Rate")
        plt.xticks(range(0, epoch_num + 1, 5))
        plt.title("Mnist-NN-Accuracy-Graph")
        plt.legend()
        if os.path.isdir("./img") == False: 
            os.mkdir("./img")
        plt.savefig("./img/Mnist-NN-Accuracy" + str(epoch_num) + ".png")
        plt.show()