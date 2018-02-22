#!/usr/bin/env python
import sacred
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, datasets

ex = sacred.Experiment("pytorch_blstm")

class myBLSTM(nn.Module):
    def __init__(self, xdim, ydim, nout, nlayers, gpu):
        super(myBLSTM, self).__init__()
        self.blstm = nn.LSTM(xdim, nout, nlayers, bidirectional=True)
        self.cnn1  = nn.Conv1d(xdim, 10, kernel_size=4, stride=2)
        self.pool  = nn.MaxPool1d(kernel_size=9)
        self.xdim = xdim
        self.ydim = ydim
        self.optimizer = None
        self.nout = nout
        self.nlayers = nlayers
        self.gpu = gpu

    def forward(self, x):
        o, _  = self.blstm(x)
        o = F.relu(self.cnn1(F.relu(o)))
        o = self.pool(o)
        B, T, _ = o.size()
        o = o.view(B, T)
        o = F.softmax(o)
        return o

    def onestep(self, x, t, mode='train'):
        if mode == 'train':
            self.train()
        else:
            self.eval()
        if self.gpu:
            x = x.cuda()
            t = t.cuda()
        x, t = Variable(x, requires_grad=True), Variable(t)
        o = self(x)
        self.optimizer.zero_grad()
        loss = F.cross_entropy(o, t)
        loss.backward()
        self.optimizer.step()
        return loss

    def set_opt(self, opttype, optconf):
        if opttype == "SGD":
            self.optimizer = optim.SGD(self.parameters(),
                                       lr=optconf['lr'],
                                       momentum=optconf['momentum'])
        elif opttype == "Adam":
            self.optimizer = optim.Adam(self.parameters(),
                                        lr=optconf['lr'])
        else:
            raise NotImplementedError
        
        
@ex.config
def config():
    train_batch_size = 32
    test_batch_size = 1024
    xdim = 28
    ydim = 28
    nlayers = 4
    log_interval = 100
    gpu = True
    kwargs = dict()
    opttype = 'Adam'
    optconf = dict(
        lr = 0.01,
        momentum = 0.9,        
    )
    seed = 1
    epoch = 10
    assert opttype in ['Adam', 'SGD'], \
        "not {} in ['Adam', 'SGD']".format(optimizer)
    
@ex.capture
def get_iterator(train_batch_size, test_batch_size, kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=train_batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader

@ex.capture
def get_model(xdim, ydim, nlayers, opttype, optconf, gpu):
    gpu = gpu if torch.cuda.is_available() else False
    m = myBLSTM(xdim, ydim, 10, nlayers, gpu)
    m.set_opt(opttype, optconf)
    if gpu:
        m.cuda()
    return m

@ex.capture
def train(_log, seed, epoch, log_interval):
    torch.manual_seed(seed)    
    train_iterator, test_iterator = get_iterator()
    m = get_model()
    for i in range(1, epoch+1):
        _log.info(f"Start Epoch [{i}]:")
        for batch_idx, (data, target) in enumerate(train_iterator):
            B, _, X, Y = data.size()
            data  = data.view(B, X, Y)
            target  = target.view(B)            
            loss = m.onestep(data, target)
            if batch_idx % log_interval == 0 :
                loss = loss.data[0]
                _log.info(f"idx[{batch_idx}], Loss :[{loss}]")

@ex.automain
def main():
    train()


