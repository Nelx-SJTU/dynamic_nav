import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class conv1d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(conv1d_bn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class conv1d_res(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=1):
        super(conv1d_res, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm1d(out_channels)

    def forward(self, x, y):
        opt = self.bn1(self.conv1(x))
        out = F.relu(opt + y)
        return opt, out


class Planning_Model(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Planning_Model, self).__init__()
        # input is [1, 1080]
        # [1, 1080] -> [64, 359]
        self.convlayer1 = conv1d_bn(in_channels=in_channels, out_channels=64, kernel_size=7, padding=1, stride=3)

        # [64, 179] -> [64, 179]
        self.convlayer2 = conv1d_bn(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        # [64, 179] -> [64, 179]
        self.convlayer3 = conv1d_res(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        # [64, 179] -> [64, 179]
        self.convlayer4 = conv1d_bn(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)

        # [64, 179] -> [64, 179]
        self.convlayer5 = conv1d_res(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)

        # [64, 359] -> [64, 179]
        self.maxpool = nn.MaxPool1d(3, stride=2)
        # [64, 22] -> [31, 10]
        self.avgpool = nn.AvgPool2d(3, stride=2)

        self.fc1 = nn.Linear(2761, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, out_channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, depth, target):

        conv1 = self.convlayer1(depth)
        max1 = self.maxpool(conv1)
        conv2 = self.convlayer2(max1)
        res1, conv3 = self.convlayer3(conv2, max1)
        conv4 = self.convlayer4(conv3)
        _, conv5 = self.convlayer5(conv4, res1)
        avg1 = self.avgpool(conv5)

        depth_fc = torch.reshape(avg1, (1, -1))
        target_fc = torch.reshape(target, (1, -1))
        fc_input = torch.concat([depth_fc, target_fc], dim=1)

        opt = self.fc1(fc_input)
        opt = self.fc2(opt)
        opt = self.fc3(opt)

        opt = self.sigmoid(opt) * 2 * np.pi

        return opt
