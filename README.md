# Python FANN2 MNIST DATABASE test

This project tests

# Quickstart

```#!bash

sudo apt-get install libfann2 python-fann2

git clone https://github.com/alexartwww/python-fann-mnist.git
cd python-fann-mnist

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O source/train-images.idx3-ubyte.gz
gunzip source/train-images.idx3-ubyte.gz

wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -O source/train-labels-idx1-ubyte.gz
gunzip source/train-labels-idx1-ubyte.gz

wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -O source/t10k-images-idx3-ubyte.gz
gunzip source/t10k-images-idx3-ubyte.gz

wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -O source/t10k-labels-idx1-ubyte.gz
gunzip source/t10k-labels-idx1-ubyte.gz
```

# Project Goals

Best result I've got:

```#!bash

$ python ./test.py net/2018-01-08-22-16-44-425055.net
Opening net: net/2018-01-08-22-16-44-425055.net
Reading labels
Reading images
Testing
Error rate = 12.86%
```
3 layers:

```#!python

layers = [784 260 10]
desired_error = 0.001
max_iterations = 60
```
