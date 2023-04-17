import numpy as np
import pandas as pd
import matplotlib as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class se_block(nn.Module):
    def __init__(self, input_channel):
        super(se_block, self).__init__()
        self.in_channel = input_channel
        self.avg2D = nn.AdaptiveAvgPool2d((1, 1))

        self.linear1 = nn.Linear(in_features=input_channel, out_features=4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=4, out_features=input_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.avg2D(x)
        x = x.view(b, self.in_channel)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x


class pre_net(nn.Module):
    def __init__(self, block, num_blocks):
        super(pre_net, self).__init__()
        self.out_channel = 64
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.out_channel, kernel_size=3, padding=1, stride=2)
        self.norm = nn.BatchNorm2d(self.out_channel)
        self.relu = nn.ReLU()

        self.in_planes = 64
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        self.se_block = se_block(128)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        b, c, _, _ = x.size()
        se_input = x
        se_output = self.se_block(se_input)
        se_output = se_output.view(b, c, 1, 1)
        x = x * se_output.expand_as(x)
        return x


class model_mul(nn.Module):
    def __init__(self, n_class):
        super(model_mul, self).__init__()
        self.pre_net = pre_net(BasicBlock, [2, 2, 2])
        self.norm1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.norm2 = nn.BatchNorm2d(256)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(256, n_class)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        input1, input2 = x
        x_1 = self.pre_net(input1)
        x_2 = self.pre_net(input2)

        x = x_1.mul(x_2)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv1(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.avg(x)
        b, c, _, _ = x.size()
        x = x.view(b, c)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    # pass
    image_input1 = torch.rand((1, 3, 224, 224))
    image_input2 = torch.rand((1, 3, 224, 224))
    model = model_mul(4)
    out = model([image_input1, image_input2])
    print('res:', out.size())
