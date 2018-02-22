# -*- cofing: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torchvision import datasets, transforms


d = datasets.MNIST('../../data', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor()
               ]))


class OverlappedMNISTDataSet(object):
    def __init__(self, dataset=d,
                 size=100,
                 lapped_num_figure=2,
                 height=100,
                 width=100,
                 rotate=False,
                 seed=0,
                 density=0.5,
                 scale=1.0,
                 overlapped=True,
    ):
        assert overlapped, f'Not implement overlapped = {oeverlapped}'
        assert 0 <= density <= 1, f'Not 0 <= density=f{density} <= 1'
        
        self.dataset = dataset
        self.height = height
        self.width = width
        self.rotate = rotate
        self.dataset_size = len(self.dataset)
        self.lapped_num_figure = lapped_num_figure
        self.size = size
        np.random.seed(seed)
        self.idxes = np.random.randint(self.dataset_size, size=(size, lapped_num_figure))
        # Caution: consume a lot of memory.        
        _,  self.figure_xdim, self.figure_ydim = self.dataset[0][0].size()        
        self.x_margin = self.figure_xdim // 2 
        self.y_margin = self.figure_ydim // 2 
        self.center_x = np.random.randint(self.x_margin, self.width - self.x_margin,
                                          size=(size, lapped_num_figure))
        self.center_y = np.random.randint(self.y_margin, self.height - self.y_margin,
                                          size=(size, lapped_num_figure ))
        self.mean = None
        self.std = None
        self.index = 0
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self):
            raise StopIteration
        _index = self.index
        self.index += 1
        return self[_index]
    
    def __getitem__(self, i):
        idxes = self.idxes[i]
        canvas = np.zeros((self.width, self.height))
        targets = []
        target_numbers = []
        clipped_value = 0
        # density check is Not implemented
        # self.check_density()
        
        for x, y, idx in zip(self.center_x[i], self.center_y[i], idxes):
            _max = np.max(self.dataset[idx][0].numpy())
            _canvas = np.zeros((self.width, self.height))
            if _max > clipped_value:
                clipped_value = _max
            _canvas[x-self.x_margin: x+self.x_margin,
                    y-self.y_margin: y+self.y_margin] = self.dataset[idx][0].numpy()
            targets.append(_canvas)
            target_numbers.append(self.dataset[idx][1])
            canvas += _canvas
        if self.std and self.mean:
            canvas = (canvas - self.mean) / np.sqrt(self.std + 10e-9)
        else:
            canvas = np.clip(canvas, 0, clipped_value)
            
        return dict(
            observation=canvas,
            targets=targets,
            target_numbers=target_numbers)

    def normalize(self):
        L = 0
        L2 = 0
        for i in self:
            L += np.sum(i['observation'])
            L2 += np.sum(i['observation']**2)

        denominator = self.size * self.width * self.height
        self.mean = L / denominator
        self.std = L2 / denominator - self.mean**2
        print(self.mean, self.std)
        
    def __len__(self):
        return self.size
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    mnist = OverlappedMNISTDataSet(d, lapped_num_figure=10)
    mnist.normalize()
    dl = DataLoader(mnist, batch_size=1)
    for i in dl:
        o = i['observation']
        t = i['targets']        
        break
    print(torch.mean(o))
    print(torch.std(o))    
    _, W, H = o.size()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(o.view(W, H))
    plt.show()

    '''
    for _t in t:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(_t.view(W, H))
        plt.show()
    '''
