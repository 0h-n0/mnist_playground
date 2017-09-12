#!/usr/bin/python3
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms

def get_data():
    DATAPATH = '../datasets/cifar-10-batches-py'
    
    file_list = ['data_batch_1',
                 'data_batch_2',
                 'data_batch_3',
                 'data_batch_4',
                 'data_batch_5',
                 'test_batch',]

    def unpickle(file):
        import pickle
        with open(file, 'rb') as f:
            dictonary = pickle.load(f, encoding='bytes')
            ## keys (batch_lable, labels, data, filename)
        return dictonary

    all_data = [unpickle(DATAPATH + '/' + i) for i in file_list]
    return all_data



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(32*32, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 10)        
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))        

        return F.softmax(x)

batch_size = 32

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **{})
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **{})


m = CNN()
m.cuda()
optimizer = optim.SGD(m.parameters(), lr=0.1, momentum=0.9)

def train(epoch):
    m.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        print(data)
        optimizer.zero_grad()
        output = m(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
train(1)
