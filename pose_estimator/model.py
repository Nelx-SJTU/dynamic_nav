import torch.nn as nn
import torch as t
from util import conv,deconv,correlate
from torch.nn.init import kaiming_normal_, constant_

class PoseNet(nn.Module):  ##   192*144 W*H
    def __init__(self,is_batchnorm=False):
        super(PoseNet,self).__init__()
        self.is_batchnorm  =is_batchnorm
        self.conv1 =conv(self.is_batchnorm,6,32, 5,2) # out: 96*72
        self.conv2 = conv(self.is_batchnorm, 32, 64, 5,2) # out: 48*36
        self.conv3 = conv(self.is_batchnorm, 64, 128, 3, 2)# out: 24*18
        self.conv4 = conv(self.is_batchnorm, 128, 256, 3, 2)  # out: 12*9
        self.conv5 = conv(self.is_batchnorm, 256, 256, 3, 2)  # out: 6*5

        self.deconv6 = deconv(256,256,(3,4)) # out: 12*9
        self.deconv7 = deconv(512,128)  #out: 24*18
        self.deconv8 = deconv(256,64)   #out: 48*36
        self.deconv9 = deconv(128,32)    #out: 96*72
        self.deconv10 = deconv(64,1)     #out: 192*144


        self.conv3_1 = nn.Sequential(
            conv(self.is_batchnorm, 441, 256, 3, 2),# out: 6*5
            conv(self.is_batchnorm, 256, 128, 3, 1, 1),  # out: 6*5
            conv(self.is_batchnorm, 128, 128, 3, 1, 1)  # out: 6*5
        )
        self.mlp_pos = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*30, 128),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Linear(128, 32),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Linear(32, 2),
        )
        self.mlp_theta = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*30, 128),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Linear(128, 32),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Linear(32, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self,x1, x2):
        #if self.training:

        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)
        out_conv4a = self.conv4(out_conv3a)

        out_conv1b = self.conv1(x2)
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)
        out_conv4b = self.conv4(out_conv3b)

        out_corr = correlate(out_conv4a,out_conv4b)
        out_conv3_1 = self.conv3_1(out_corr)

        pos_tmp = self.mlp_pos(out_conv3_1)
        pos = pos_tmp.view(pos_tmp.size(0),-1)
        theta_tmp  =self.mlp_theta(out_conv3_1)
        theta = theta_tmp.view(theta_tmp.size(0),-1)

        out_conv5 = self.conv5(out_conv4b)

        out_deconv6 = self.deconv6(out_conv5)

        in_deconv7 = t.cat([out_deconv6,out_conv4b],dim=1)
        out_deconv7 = self.deconv7(in_deconv7)


        in_deconv8 = t.cat([out_deconv7,out_conv3b],dim=1)
        out_deconv8 = self.deconv8(in_deconv8)

        in_deconv9 = t.cat([out_deconv8,out_conv2b],dim=1)
        out_deconv9 = self.deconv9(in_deconv9)

        in_deconv10 = t.cat([out_deconv9,out_conv1b],dim=1)
        depth = self.deconv10(in_deconv10)

        return depth ,pos,theta

