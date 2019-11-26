# Residual block architecture

import torch
import torch.nn as nn 
from torch.nn import functional as F
from torch.autograd import Variable

import pdb

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = (kernel_size-1) // 2
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv3d(x)
        return out

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample_mode, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample_mode = upsample_mode
        self.upsample = upsample
        padding = (kernel_size-1) // 2
        if upsample_mode == "lr" and upsample:
            self.conv3d = nn.Conv3d(in_channels, out_channels * upsample * upsample * upsample, kernel_size, stride,
                                    padding)
            self.voxel_shuffle = VoxelShuffle(upsample)
        else:
            self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)


    def forward(self, x):
        if self.upsample:
            if self.upsample_mode == "hr":
                x = F.interpolate(x, mode='nearest', scale_factor=self.upsample)
                out = self.conv3d(x)
            elif self.upsample_mode == "lr":
                x = self.conv3d(x)
                out = self.voxel_shuffle(x)
        else:
            out = self.conv3d(x)

        return out

class VoxelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(VoxelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        batch_size, c, h, w, l = input.size()
        rh, rw, rl = self.upscale_factor, self.upscale_factor, self.upscale_factor
        oh, ow, ol = h * rh, w * rw, l * rl
        oc = c // (rh * rw * rl)
        input_view = input.contiguous().view(
            batch_size, rh, rw, rl, oc, h, w, l
        )
        shuffle_out = input_view.permute(0, 4, 5, 1, 6, 2, 7, 3).contiguous()
        out = shuffle_out.view(
            batch_size, oc, oh, ow, ol
        )
        return out

class ForwardBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 downsample_factor=2):
        super(ForwardBlockGenerator, self).__init__()
        self.relu = nn.ReLU()

        self.p1_conv0 = ConvLayer(in_channels, out_channels, kernel_size, downsample_factor)
        self.p1_in0 = nn.InstanceNorm3d(out_channels, affine=True)
        self.p1_conv1 = ConvLayer(out_channels, out_channels, kernel_size, stride)
        self.p1_in1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.p1_conv2 = ConvLayer(out_channels, out_channels, kernel_size, stride)
        self.p1_in2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.p1_conv3 = ConvLayer(out_channels, out_channels, kernel_size, stride)
        self.p1_in3 = nn.InstanceNorm3d(out_channels, affine=True)

        self.p2_conv0 = ConvLayer(in_channels, out_channels, kernel_size, downsample_factor)
        self.p2_in0 = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x, norm):
        out = self.p1_conv0(x)
        if norm == "Instance":
            pdb.set_trace()
            out = self.p1_in0(out)
        out = self.relu(out)

        out = self.p1_conv1(out)
        if norm == "Instance":
            out = self.p1_in1(out)
        out = self.relu(out)

        out = self.p1_conv2(out)
        if norm == "Instance":
            out = self.p1_in2(out)
        out = self.relu(out)

        out = self.p1_conv3(out)
        if norm == "Instance":
            out = self.p1_in3(out)
        out = self.relu(out)

        residual = self.p2_conv0(x)
        if norm == "Instance":
            residual = self.p2_in0(residual)
        residual = self.relu(residual)

        out = out + residual
        return out

class BackwardBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, upsample_mode="lr",
                 upsample_factor=2):
        super(BackwardBlockGenerator, self).__init__()
        self.relu = nn.ReLU()

        self.p1_in0 = nn.InstanceNorm3d(in_channels, affine=True)
        self.p1_conv0 = UpsampleConvLayer(in_channels, in_channels, kernel_size, stride, upsample_mode)
        self.p1_in1 = nn.InstanceNorm3d(in_channels, affine=True)
        self.p1_conv1 = UpsampleConvLayer(in_channels, in_channels, kernel_size, stride, upsample_mode)
        self.p1_in2 = nn.InstanceNorm3d(in_channels, affine=True)
        self.p1_conv2 = UpsampleConvLayer(in_channels, in_channels, kernel_size, stride, upsample_mode)
        self.p1_in3 = nn.InstanceNorm3d(in_channels, affine=True)
        self.p1_conv3 = UpsampleConvLayer(in_channels, out_channels, kernel_size, stride, upsample_mode,
                                          upsample=upsample_factor)

        self.p2_in0 = nn.InstanceNorm3d(in_channels, affine=True)
        self.p2_conv0 = UpsampleConvLayer(in_channels, out_channels, kernel_size, stride, upsample_mode,
                                          upsample=upsample_factor)

    def forward(self, x, norm):
        out = x
        if norm == "Instance":
            out = self.p1_in0(out)
        out = self.relu(out)
        out = self.p1_conv0(out)

        if norm == "Instance":
            out = self.p1_in1(out)
        out = self.relu(out)
        out = self.p1_conv1(out)

        if norm == "Instance":
            out = self.p1_in2(out)
        out = self.relu(out)
        out = self.p1_conv2(out)

        if norm == "Instance":
            out = self.p1_in3(out)
        out = self.relu(out)
        out = self.p1_conv3(out)

        residual = x
        if norm == "Instance":
            residual = self.p2_in0(residual)
        residual = self.relu(residual)
        residual = self.p2_conv0(residual)

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



