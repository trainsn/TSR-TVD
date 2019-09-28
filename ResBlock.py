# Residual block architecture 

import torch
import torch.nn as nn 
from torch.nn import functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv3d(x)
        return out

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        padding = kernel_size // 2
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, mode='nearest', scale_factor=self.upsample)
        out = self.conv3d(x)
        return out

class ForwardBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 downsample_factor=2):
        super(ForwardBlockGenerator, self).__init__()
        self.relu = nn.ReLU()

        self.p1_conv0 = ConvLayer(in_channels, out_channels, kernel_size, stride)
        self.p1_conv1 = ConvLayer(out_channels, out_channels, kernel_size, stride)
        self.p1_conv2 = ConvLayer(out_channels, out_channels, kernel_size, stride)
        self.p1_conv3 = ConvLayer(out_channels, out_channels, kernel_size, downsample_factor)

        self.p2_conv0 = ConvLayer(in_channels, out_channels, kernel_size, downsample_factor)


    def forward(self, x):
        out = self.relu(self.p1_conv0(x))
        out = self.relu(self.p1_conv1(out))
        out = self.relu(self.p1_conv2(out))
        out = self.p1_conv3(out)

        residual = self.p2_conv0(x)

        out = out + residual
        return out

class BackwardBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 upsample_factor=2):
        super(BackwardBlockGenerator, self).__init__()
        self.relu = nn.ReLU()

        self.p1_conv0 = UpsampleConvLayer(in_channels, in_channels, kernel_size, stride, upsample=upsample_factor)
        self.p1_conv1 = UpsampleConvLayer(in_channels, in_channels, kernel_size, stride)
        self.p1_conv2 = UpsampleConvLayer(in_channels, in_channels, kernel_size, stride)
        self.p1_conv3 = UpsampleConvLayer(in_channels, out_channels, kernel_size, stride)

        self.p2_conv0 = UpsampleConvLayer(in_channels, out_channels, kernel_size, stride, upsample=upsample_factor)

    def forward(self, x):
        out = self.relu(self.p1_conv0(x))
        out = self.relu(self.p1_conv1(out))
        out = self.relu(self.p1_conv1(out))
        out = self.p1_conv3(out)

        residual = self.p2_conv0(x)

        out = out + residual
        return out




