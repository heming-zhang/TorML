if [ -d MNIST ]; then
    echo "MNIST directory already exists, exiting."
    exit 1
fi

if [ -d MNIST_BIN ]; then
    echo "MNIST_BIN directory already exists, exiting."
    exit 1
fi

mkdir MNIST
wget -P ./MNIST/ http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz \
http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz \
http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz \
http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz \

pushd MNIST
gunzip *
popd

mkdir MNIST_BIN
hexdump -C ./MNIST/train-images-idx3-ubyte >./MNIST_BIN/trainimages
hexdump -C ./MNIST/train-labels-idx1-ubyte >./MNIST_BIN/trainlabels
hexdump -C ./MNIST/t10k-images-idx3-ubyte >./MNIST_BIN/testimages
hexdump -C ./MNIST/t10k-labels-idx1-ubyte >./MNIST_BIN/testlabels