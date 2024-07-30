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


class resnet18(nn.Module):
    def __init__(self, NumClass=10) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.layer11 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.layer12 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
        )

        self.layer21 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128)
        )
        self.layer22 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128)
        )
        self.downsample2 = nn.Conv2d(64, 128, 1, 2, 0)

        self.layer31 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256)
        )
        self.layer32 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256)
        )
        self.downsample3 = nn.Conv2d(128, 256, 1, 2, 0)

        self.layer41 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512)
        )
        self.layer42 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512)
        )
        self.downsample4 = nn.Conv2d(256, 512, 1, 2, 0)

        self.avgpool = nn.AvgPool2d(7, 1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, NumClass)

    def forward(self, x):
        x = self.downsample(x)
        x = self.layer11(x) + x
        x = self.layer12(x) + x
        x = self.layer21(x) + self.downsample2(x)
        x = self.layer22(x) + x
        x = self.layer31(x) + self.downsample3(x)
        x = self.layer32(x) + x
        x = self.layer41(x) + self.downsample4(x)
        x = self.layer42(x) + x
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    print('models.py')
    mdl = resnet18()
    X = np.random.random((2, 3, 224, 224)).astype(np.float32)
    X = torch.tensor(X)
    X = mdl(X)
    print('output', X.shape)
