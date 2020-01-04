import os
from mnist_parse import ParseFile
from mnist_nn import RunMnistNN
from mnist_nn import MnistNNLoad
from mnist_nn import MnistParameter

def parse_file():
    train_img = "./MNIST/train-images-idx3-ubyte"
    train_label = "./MNIST/train-labels-idx1-ubyte"
    test_img = "./MNIST/t10k-images-idx3-ubyte"
    test_label = "./MNIST/t10k-labels-idx1-ubyte"
    ParseFile(train_img, train_label, 0).parse_image()
    ParseFile(train_img, train_label, 0).parse_label()
    ParseFile(test_img, test_label, 0).parse_image()
    ParseFile(test_img, test_label, 0).parse_label()

def classify_file():
    train_img = "./MNIST/train-images-idx3-ubyte"
    train_label = "./MNIST/train-labels-idx1-ubyte"
    test_img = "./MNIST/t10k-images-idx3-ubyte"
    test_label = "./MNIST/t10k-labels-idx1-ubyte"
    for i in range(0, 10):
        ParseFile(train_img, train_label, i).classify()
        ParseFile(test_img, test_label, i).classify()

def plot(epoch, train_loss, test_loss):
    return 0

if __name__ == "__main__":
    if os.path.isdir("./mnist") == False:
        parse_file()
    if os.path.isdir("./mnist_classified") == False:
        classify_file()
    # Load data with mini-batch
    nn_batchsize = 640
    nn_worker = 2
    train_loader, test_loader = MnistNNLoad(nn_batchsize, 
                nn_worker).load_data()
    # Use Neural Network as training model
    nn_learning_rate = 0.01
    nn_momentum = 0.5
    model, criterion, optimizer = MnistParameter(
                nn_learning_rate,
                nn_momentum).mnist_function()
    # Get experimental results in every epoch
    nn_epoch_num = 20
    for epoch in range(1 : nn_epoch_num + 1):
        train_loss = RunMnistNN(model,
                    criterion,
                    optimizer,
                    train_loader,
                    test_loader).train_mnist_nn()
        test_loss = RunMnistNN(model,
                    criterion,
                    optimizer,
                    train_loader,
                    test_loader).test_mnist_nn()
        print(epoch, train_loss, test_loss)