import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class conv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(conv2d_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class deconv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(deconv2d_bn, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                        stride=stride, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class Mapping_Model(nn.Module):
    def __init__(self, in_channels=3):
        super(Mapping_Model, self).__init__()
        # input : [in_channels, 224, 224]
        # output : [16, 112, 112]
        self.convlayer1 = conv2d_bn(in_channels=in_channels, out_channels=16, kernel_size=4, padding=1, stride=2)

        # output : [64, 56, 56]
        self.convlayer2 = conv2d_bn(in_channels=16, out_channels=64, kernel_size=4, padding=1, stride=2)

        # output : [256, 28, 28]
        self.convlayer3 = conv2d_bn(in_channels=64, out_channels=256, kernel_size=4, padding=1, stride=2)

        # output : [1024, 14, 14]
        self.convlayer4 = conv2d_bn(in_channels=256, out_channels=1024, kernel_size=4, padding=1, stride=2)

        # output : [2048, 7, 7]
        self.convlayer5 = conv2d_bn(in_channels=1024, out_channels=2048, kernel_size=4, padding=1, stride=2)

        # output : [1,1024,14,14]  (after torch.cat : [1,2048,14,14] )
        self.deconv1 = deconv2d_bn(in_channels=2048, out_channels=1024, kernel_size=2, stride=2)
        # output : [1,1024,14,14]
        self.conv1 = conv2d_bn(in_channels=2048, out_channels=1024, kernel_size=3, stride=1, padding=1)

        # output : [1,256,28,28]  (after torch.cat : [1,512,28,28] )
        self.deconv2 = deconv2d_bn(in_channels=1024, out_channels=256, kernel_size=2, stride=2)
        # output : [1,256,28,28]
        self.conv2 = conv2d_bn(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)

        # output : [1,64,56,56]  (after torch.cat : [1,128,56,56] )
        self.deconv3 = deconv2d_bn(in_channels=256, out_channels=64, kernel_size=2, stride=2)
        # output : [1,64,56,56]
        self.conv3 = conv2d_bn(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)

        # output : [1,16,112,112]  (after torch.cat : [1,32,112,112] )
        self.deconv4 = deconv2d_bn(in_channels=64, out_channels=16, kernel_size=2, stride=2)
        # output : [1,16,112,112]
        self.conv4 = conv2d_bn(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)

        # output : [1,1,224,224]
        self.deconv5 = deconv2d_bn(in_channels=16, out_channels=1, kernel_size=2, stride=2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1 = self.convlayer1(x)
        c2 = self.convlayer2(c1)
        c3 = self.convlayer3(c2)
        c4 = self.convlayer4(c3)
        x = self.convlayer5(c4)

        dc1 = self.deconv1(x)
        x = torch.cat([dc1, c4], dim=1)
        x = self.conv1(x)

        dc2 = self.deconv2(x)
        x = torch.cat([dc2, c3], dim=1)
        x = self.conv2(x)

        dc3 = self.deconv3(x)
        x = torch.cat([dc3, c2], dim=1)
        x = self.conv3(x)

        dc4 = self.deconv4(x)
        x = torch.cat([dc4, c1], dim=1)
        x = self.conv4(x)

        x = self.deconv5(x)

        x = self.sigmoid(x)

        return x
