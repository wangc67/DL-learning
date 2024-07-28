import torch.nn as nn
import torch
import numpy as np


class convnet(nn.Module):
    def __init__(self, NumClass=10):
        super().__init__()
        self.conv_layer_1 = torch.nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv_layer_2 = torch.nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv_layer_3 = torch.nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv_layer_4 = torch.nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.hidden_layer = nn.Linear(18432, 2000)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(2000, NumClass)

    def forward(self, ip):
        output = self.conv_layer_1(ip)
        output = self.conv_layer_2(output)
        output = self.conv_layer_3(output)
        output = self.conv_layer_4(output)
        output = self.flatten(output)
        # print(output.shape)
        output = self.hidden_layer(output)
        output = self.relu(output)
        output = self.output_layer(output) 
        return output


class cnn_cifar(nn.Module):
    def __init__(self, NumClass=100):
        super().__init__()
        self.conv_layer_1 = torch.nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv_layer_2 = torch.nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv_layer_3 = torch.nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv_layer_4 = torch.nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.hidden_layer = nn.Linear(512, 200)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(200, NumClass)

    def forward(self, ip):
        output = self.conv_layer_1(ip)
        output = self.conv_layer_2(output)
        output = self.conv_layer_3(output)
        output = self.conv_layer_4(output)
        output = self.flatten(output)
        # print(output.shape)
        output = self.hidden_layer(output)
        output = self.relu(output)
        output = self.output_layer(output) 
        return output


if __name__ == '__main__':
    print('models.py')
    mdl = convnet()
    X = np.random.random((2, 3, 200, 200)).astype(np.float32)
    X = torch.tensor(X)
    X = mdl(X)
    print(X.shape)
