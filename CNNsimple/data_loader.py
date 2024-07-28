import os
import torch
import numpy as np
from random import shuffle
import torch.utils
from torch.utils.data import DataLoader
from PIL import Image
import json
import torch.utils.data
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms

'''
https://zhuanlan.zhihu.com/p/340465632
https://geek-docs.com/pytorch/pytorch-questions/47_pytorch_pytorch_how_to_use_dataloaders_for_custom_datasets.html
'''

class dataset():
    def __init__(self, path, rotate=False, flip=False, clip=True):
        self.path = path
        self.GetDataPath() # load self.data_path: list[tuple[str, int]]
        self.len = len(self.data_path)
        self.rotate = rotate
        self.flip = flip
        self.clip = clip

        self.xx = 200
        self.yy = 200 # pic size not same, clip to 

    def __getitem__(self, index):
        return self.process(self.data_path[index])

    def __len__(self):
        return self.len

    def GetDataPath(self):
        flag = 0
        if 'val' in self.path or 'train' in self.path:
            flag = 1
        elif 'test' in self.path:
            flag = 0
            with open('data\\10_plants_dataset\\test_labels.json') as ff:
                self.test_label = json.load(ff)
        
        with open('data\\10_plants_dataset\\label_idx.json') as f:
            self.label_idx = json.load(f)
        
        self.data_path = []
        for base_dir, dir, files in os.walk(self.path):
            for item in files:
                if self.is_picture(item):
                    file_dir = os.path.join(base_dir, item)
                    if flag == 1:
                        label = self.label_idx[file_dir.split('\\')[-2]]
                    else:
                        idx = file_dir.split('\\')[-1].split('.')[0]
                        label = self.label_idx[self.test_label[idx]]
                    self.data_path.append([file_dir, label])

    def is_picture(self, img: str):
        if '.jpg' in img:
            return True
        else:
            return False
        
    def process(self, item: tuple[str, int]):
        img, label = Image.open(item[0]), item[1]
        if self.rotate:
            theta = np.random.uniform(0., 2*np.pi)
            img = img.rotate(theta)
        if self.clip: # vvvvvvvvvvvvv
            X, Y = img.size
            x0 = np.random.randint(0, X - self.xx)
            y0 = np.random.randint(0, Y - self.yy)
            img = img.crop((x0, y0, x0 + self.xx, y0 + self.yy))
        else:         # ^^^^^^^^^^^^^
            img = img.resize((self.xx, self.yy))
        if self.flip:
            lr, tb = np.random.randint(0, 2), np.random.randint(0, 2)
            if lr == 1:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if tb == 1:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = torch.tensor(np.array(img, dtype=np.float32))
        img = img.permute(2, 0, 1)
        return (img, label)



path = '../data/10_plants_dataset'


def GetDataLoader(batch_size: int, mode='train'):
    '''
    mode == 'train': return train, val;
    else return test
    '''
    TrainDataset = dataset(path + '/train', rotate=True, flip=True, clip=False)
    ValDataset = dataset(path + '/val')
    TestDataset = dataset(path + '/test')
    if mode == 'train':
        TrainLoader = DataLoader(TrainDataset, batch_size, shuffle=True, pin_memory=True)
        ValLoader = DataLoader(ValDataset, batch_size, shuffle=False, pin_memory=True)
        return TrainLoader, ValLoader
    else:
        TestLoader = DataLoader(TestDataset, batch_size, shuffle=False, pin_memory=True)
        return TestLoader

def GetCIFAR100(batch_size: int, mode='train'):
    transform = transforms.Compose([transforms.ToTensor(),
                                # transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                # transforms.RandomHorizontalFlip(),
                                # transforms.RandomRotation(2.8),
                                # transforms.RandomGrayscale(0.2)
                                ])
    if mode == 'train':
        val_ratio = 0.2
        data = CIFAR100('../data', train=True, transform=transform)
        train_data, val_data = torch.utils.data.random_split(data, [int((1 - val_ratio) * len(data)), int(val_ratio * len(data))])
        TrainLoader = DataLoader(train_data, batch_size, shuffle=True, pin_memory=True)
        ValLoader = DataLoader(val_data, batch_size, shuffle=False, pin_memory=True)
        return TrainLoader, ValLoader
    else:
        test_data =  CIFAR100('../data', train=False, transform=transform)
        TestLoader = DataLoader(test_data, batch_size, shuffle=False, pin_memory=True)
        return TestLoader

if __name__ == '__main__':
    print('data_loader.py')
    
    minx, miny = 1213456, 1256564

    aa = dataset(path + '/train', clip=True)
    for i in range(aa.len):
        _, x, y = aa.__getitem__(i)[0].shape
        print(x, y)
        input()
        minx = min(x, minx)
        miny = min(y, miny)
    print(f'min {minx} {miny}')
    # input()
    aa = dataset(path + '/val', clip=False)
    for i in range(aa.len):
        _, x, y = aa.__getitem__(i)[0].shape
        minx = min(x, minx)
        miny = min(y, miny)
    print(f'min {minx} {miny}')
    aa = dataset(path + '/test', clip=False)
    for i in range(aa.len):
        _, x, y = aa.__getitem__(i)[0].shape
        minx = min(x, minx)
        miny = min(y, miny)
        print(x / y)
    print(f'min {minx} {miny}')
    
    
    