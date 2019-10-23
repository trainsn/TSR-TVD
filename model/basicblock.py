# Residual block architecture

import torch
import torch.nn as nn 
from torch.nn import functional as F
from torch.autograd import Variable

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

        # out = out + residual
        out = residual
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

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, stride):
        super(ConvLSTMCell, self).__init__()

        padding = kernel_size // 2

        self.Wxf = nn.Conv3d(input_channels, hidden_channels, kernel_size, stride, padding, bias=True)
        self.Whf = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, stride, padding, bias=False)
        self.Wxi = nn.Conv3d(input_channels, hidden_channels, kernel_size, stride, padding, bias=True)
        self.Whi = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, stride, padding, bias=False)
        self.Wxo = nn.Conv3d(input_channels, hidden_channels, kernel_size, stride, padding, bias=True)
        self.Who = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, stride, padding, bias=False)
        self.Wxc = nn.Conv3d(input_channels, hidden_channels, kernel_size, stride, padding, bias=True)
        self.Whc = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, stride, padding, bias=False)

    def forward(self, x, h0, c0):
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h0))
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h0))
        o = torch.sigmoid(self.Wxo(x) + self.Who(h0))
        c = i * torch.tanh(self.Wxc(x) + self.Whc(h0)) + f * c0
        h = o * torch.tanh(c)
        return h, c

    def init_hidden(self, batch_size, hidden_channels, shape):
        return (Variable(torch.zeros(batch_size, hidden_channels, shape[0], shape[1], shape[2])).cuda(),
                Variable(torch.zeros(batch_size, hidden_channels, shape[0], shape[1], shape[2])).cuda())



