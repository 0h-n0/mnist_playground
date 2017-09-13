#!/usr/bin/env python3
import time
import importlib
from pathlib import Path
import urllib.request


from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.datasets import fetch_mldata
import numpy as np
import mxnet as mx
import tensorboard as tb

ex = Experiment('mnist_blstm')
ex.observers.append(FileStorageObserver.create('results'))


def nn(data):
    flatten = mx.sym.flatten(data=data)
    nn1 = mx.sym.FullyConnected(data=flatten, num_hidden=128)
    nn2 = mx.sym.Activation(data=nn1, act_type='relu')    
    nn3 = mx.sym.FullyConnected(data=nn2, num_hidden=64)
    nn4 = mx.sym.Activation(data=nn3, act_type='relu')
    nn5 = mx.sym.FullyConnected(data=nn4, num_hidden=10)
    out = mx.sym.SoftmaxOutput(data=nn5, name='softmax')
    return out

def cnn(data):
    nn = mx.sym.Convolution(data=data, kernel=(2, 2), num_filter=20)
    nn = mx.sym.Activation(data=nn, act_type='relu')
    nn = mx.sym.Pooling(data=nn, pool_type='max', kernel=(2, 2), stride=(1,1))
    nn = mx.sym.flatten(data=nn)
    nn = mx.sym.Activation(data=nn, act_type='relu')
    nn = mx.sym.FullyConnected(data=nn, num_hidden=10)
    out = mx.sym.SoftmaxOutput(data=nn, name='softmax')
    return out

def rnn(data):
    flatten = mx.sym.flatten(data=data)
    
    out = mx.sym.SoftmaxOutput(data=nn, name='softmax')
    return nn

def acc(model, data, labels):
    model.fowa
    
@ex.config
def cfg():
    path_to_data = '../data/mnist' # set data path.
    path_to_results = './results'  # set results path.
    num_train = 60000              # set the number of train samples.
    batch = 10 # set batch size
    epoch = 1 # set epoch
    context = 'cpu' # set background archtecture
    ngpu = 1 # set the number of gpu
    model = 'nn' # set the name of model.
    background = 'mnxet' # set the name of deep learning library
    gpuid = 0
    lr = 0.1
    block = 60000

@ex.command
def graph(model, path_to_results):
    data = mx.sym.var('data')
    if model == 'nn':
        net = nn(data)
    elif model == 'cnn':
        net = cnn(data)        
    elif model == 'rnn':
        net = rnn(data)
    a = mx.viz.plot_network(net)
    p = Path(path_to_results)
    filename = model
    out = p / filename
    a.render(out.resolve())
    

@ex.capture
def show_cfg(_log, _seed, path_to_data, path_to_results, lr,
             num_train, batch, epoch, context, ngpu, model, background):
    _log.info('path: {}'.format(path_to_data))
    _log.info('path: {}'.format(path_to_results))    
    _log.info('random seed: {}'.format(_seed))
    _log.info('batch: {}'.format(batch))
    _log.info('epoch: {}'.format(epoch))
    _log.info('context: {}'.format(context))                
    _log.info('ngpu: {}'.format(ngpu))
    _log.info('model: {}'.format(model))
    _log.info('Learning rate: {}'.format(lr))    
    _log.info('background: {}'.format(background))        

    
@ex.capture
def run(_log, _seed, path_to_data, path_to_results, num_train, batch, epoch,
        context, ngpu, gpuid, lr, model):
    if context == 'cpu':
        mxcontext = mx.cpu()
    elif context == 'gpu':
        mxcontext = mx.gpu(gpuid)
        
    _log.info('Start train.')

    _log.info('Downloading MNIST DATA.')
    p = Path(path_to_data)
    mnist = fetch_mldata('MNIST original', data_home=p.resolve())
    _log.info('Completed downloading MNIST DATA.')

    np.random.seed(_seed)
    random_seq = np.arange(70000)
    np.random.shuffle(random_seq)

    ## shuffle data
    train_data = mnist.data[random_seq[:num_train]]
    train_labels = mnist.target[random_seq[:num_train]]
    test_data = mnist.data[random_seq[num_train:]]
    test_labels = mnist.target[random_seq[num_train:]]
    ### 
    
    data = mx.sym.var('data')
    if model == 'nn':
        dnnarch = nn(data)
    elif model == 'cnn':
        dnnarch = cnn(data)        
    elif model == 'rnn':
        dnnarch = rnn(data)
        
    model = mx.mod.Module(symbol=dnnarch, data_names=['data'],
                          label_names=['softmax_label'], context=mxcontext)


    loss = mx.metric.create('loss')
    acc = mx.metric.Accuracy()

    train_iter = mx.io.NDArrayIter(data={'data':mx.nd.array(train_data)},
                               label={'softmax_label':mx.nd.array(train_labels)},
                               batch_size=batch)
    
    val_iter = mx.io.NDArrayIter(data={'data':mx.nd.array(test_data)},
                                   label={'softmax_label':mx.nd.array(test_labels)},
                                   batch_size=100)

    mnist = mx.test_utils.get_mnist()
    train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch, shuffle=True)
    val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch)

    
    model.bind(data_shapes=train_iter.provide_data,
               label_shapes=train_iter.provide_label)
    model.init_params(initializer=mx.init.Xavier(magnitude=2.))
    
    model.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', lr), ))
    acc = mx.metric.Accuracy()
    loss = mx.metric.Loss()    
    cvacc = mx.metric.Accuracy()
    cvloss = mx.metric.Loss()    

    for i in range(epoch):
        _log.info('epoch:{:2d} {}'.format(i+1, '='*20+'>'))
        loss.reset()
        acc.reset()
        train_iter.reset()
        val_iter.reset()        
        s = time.time()
        for b in train_iter:
            model.forward(b, is_train=True)
            model.update_metric(loss, b.label)
            model.update_metric(acc, b.label)
            model.backward()
            model.update()
        _log.info('epoch:{:2d}, Train {}: {}'.format(i+1, *loss.get()))        
        _log.info('epoch:{:2d}, Train {}: {}'.format(i+1, *acc.get()))
        _log.info('epoch:{:2d}, Time : {:6.4f}(sec)'.format(i+1,time.time() - s))
        s = time.time()
        for b in val_iter:
            model.forward(b, is_train=False)
            model.update_metric(cvloss, b.label)
            model.update_metric(cvacc, b.label)
            model.backward()
            model.update()

        _log.info('epoch:{:2d}, CV {}: {}'.format(i+1, *cvloss.get()))        
        _log.info('epoch:{:2d}, CV {}: {}'.format(i+1, *cvacc.get()))
        _log.info('epoch:{:2d}, Time : {:6.4f}(sec)'.format(i+1,time.time() - s))
        
            
@ex.automain
def main():
    show_cfg()
    run()



