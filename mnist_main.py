from mnist_parse import ParseFile
import os

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

if __name__ == "__main__":
    if os.path.isdir("./mnist") == False:
        parse_file()
    if os.path.isdir("./mnist_classified") == False:
        classify_file()