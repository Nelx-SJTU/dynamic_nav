import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import math
import torch.utils.model_zoo as model_zoo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class conv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(conv2d_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class deconv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(deconv2d_bn, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class Topo_Judge_Model(nn.Module):
    def __init__(self, in_channels=3):
        super(Topo_Judge_Model, self).__init__()
        # Write yout model here

    def forward(self, x):
        topo_change_mode = 0
        action_info = " "

        # Write yout model here

        return topo_change_mode, action_info