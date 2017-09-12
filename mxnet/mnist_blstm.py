#!/usr/bin/env python3
import struct
from pathlib import Path
import urllib.request

from sklearn.datasets import fetch_mldata
import numpy as np
import mxnet as mx
    

path_to_data = '../data/mnist'
p = Path(path_to_data)

mnist = fetch_mldata('MNIST original', data_home=p.resolve())


print(mnist.target[0])
print(len(mnist.target))

np.random.seed(1)
random_seq = np.arange(70000)
num_train = 60000
np.random.shuffle(random_seq)

## shuffle data
train_data = mnist.data[random_seq[:num_train]]
train_labels = mnist.target[random_seq[:num_train]]
test_data = mnist.data[random_seq[num_train:]]
test_labels = mnist.target[random_seq[num_train:]]

### 
def nn(data):
    flatten = mx.sym.flatten(data=data)
    fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=128)
    act1 = mx.sym.Activation(data=fc1, act_type='relu')
    fc2 = mx.sym.FullyConnected(data=act1, num_hidden=10)
    act2 = mx.sym.Activation(data=fc2, act_type='relu')
    out = mx.sym.Softmax(data=act2, name='softmax')
    return out

batch = 10
epoch = 2
    
data = mx.sym.var('data')
model = mx.mod.Module(symbol=nn(data), data_names=['data'],
                      label_names=['softmax_label'], context=mx.gpu(0))

model.bind(data_shapes=[('data', (batch, 784))],
           label_shapes=[('softmax_label', (batch, 10))])

model.init_params(initializer=mx.init.Xavier(magnitude=2.))
model.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
metric = mx.metric.create('acc')


for i in range(epoch):
    for j in range(0, num_train, batch):
        print(train_data[j])
        #data, targets = generator(batch, labelsize, dtype='l')
        #_list = []
