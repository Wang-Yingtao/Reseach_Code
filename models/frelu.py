import torch
from torch import nn
from models.conv2d_mtl import Conv2dMtl

class FReLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.Conv2d = Conv2dMtl
        # self.conv1 = self.Conv2d(3, iChannels, kernel_size=3, stride=1, padding=1)

        self.depthwise_conv_bn = nn.Sequential(
            self.Conv2d(in_channel, in_channel, 3, padding=1, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel))

    def forward(self, x):
        funnel_x = self.depthwise_conv_bn(x)
        return torch.max(x, funnel_x)
