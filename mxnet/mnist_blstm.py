#!/usr/bin/env python3
import struct

from pathlib import Path
import urllib.request

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.datasets import fetch_mldata
import numpy as np
import mxnet as mx

ex = Experiment('mnist_blstm')
ex.observers.append(FileStorageObserver.create('results'))


def nn(data):
    flatten = mx.sym.flatten(data=data)
    fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=128)
    act1 = mx.sym.Activation(data=fc1, act_type='relu')
    fc2 = mx.sym.FullyConnected(data=act1, num_hidden=10)
    act2 = mx.sym.Activation(data=fc2, act_type='relu')
    out = mx.sym.Softmax(data=act2, name='softmax')
    return out

@ex.config
def cfg():
    path_to_data = '../data/mnist' # set data path.
    path_to_results = './results'  # set results path.
    num_train = 60000              # set the number of train samples.
    batch = 10 # set batch size
    epoch = 2 # set epoch
    context = 'cpu' # set background archtecture
    ngpu = 1 # set the number of gpu
    model = 'nn' # set the name of model.
    background = 'mnxet' # set the name of deep learning library

@ex.capture
def show_cfg(_log, _seed, path_to_data, path_to_results,
             num_train, batch, epoch, context, ngpu, model, background):
    _log.info('path: {}'.format(path_to_data))
    _log.info('path: {}'.format(path_to_results))    
    _log.info('random seed: {}'.format(_seed))
    _log.info('batch: {}'.format(batch))
    _log.info('epoch: {}'.format(epoch))
    _log.info('context: {}'.format(context))                
    _log.info('ngpu: {}'.format(ngpu))
    _log.info('model: {}'.format(model))
    _log.info('background: {}'.format(background))        

    
@ex.capture
def run(_log, _seed, path_to_data, path_to_results, num_train, batch, epoch,
        context, ngpu):
    
    _log.info('Start train.')

    _log.info('Downloading MNIST DATA.')
    p = Path(path_to_data)
    mnist = fetch_mldata('MNIST original', data_home=p.resolve())
    _log.info('Completed downloading MNIST DATA.')

    np.random.seed(_seed)
    random_seq = np.arange(70000)
    num_train = 60000
    np.random.shuffle(random_seq)

    ## shuffle data
    train_data = mnist.data[random_seq[:num_train]]
    train_labels = mnist.target[random_seq[:num_train]]
    test_data = mnist.data[random_seq[num_train:]]
    test_labels = mnist.target[random_seq[num_train:]]
    ### 
    
    data = mx.sym.var('data')
    model = mx.mod.Module(symbol=nn(data), data_names=['data'],
                          label_names=['softmax_label'], context=mx.cpu())

    model.bind(data_shapes=[('data', (batch, 784))],
           label_shapes=[('softmax_label', (batch, 10))])

    model.init_params(initializer=mx.init.Xavier(magnitude=2.))
    model.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
    metric = mx.metric.create('acc')

    for i in range(epoch):
        for j in range(0, num_train - batch, batch):
            x = train_data[j:j+batch]
            y = train_labels[j:j+batch]
            model(x)
            
@ex.automain
def main():
    show_cfg()
    run()





