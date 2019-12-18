# Discriminator architecture

import torch
import torch.nn as nn

from basicblock import ConvLayer
import pdb

class Discriminator(nn.Module):
    def __init__(self, dis_sn):
        super(Discriminator, self).__init__()

        # volume classification subnet
        self.conv1 = ConvLayer(in_channels=1, out_channels=64, sn=dis_sn, kernel_size=4, stride=2)
        self.conv2 = ConvLayer(in_channels=64, out_channels=128, sn=dis_sn, kernel_size=4, stride=2)
        self.conv3 = ConvLayer(in_channels=128, out_channels=256, sn=dis_sn, kernel_size=4, stride=2)
        self.conv4 = ConvLayer(in_channels=256, out_channels=512, sn=dis_sn, kernel_size=4, stride=2)
        self.conv5 = nn.Conv3d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0)
        if dis_sn:
            self.conv5 = nn.utils.spectral_norm(self.conv5, eps=1e-4)

        self.leakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        outputs = []
        for i in range(x.shape[1]):
            # pdb.set_trace()
            out = self.leakyReLU(self.conv1(x[:, i, :]))
            out = self.leakyReLU(self.conv2(out))
            out = self.leakyReLU(self.conv3(out))
            out = self.leakyReLU(self.conv4(out))
            out = self.conv5(out)
            outputs.append(out.unsqueeze(1))
        outputs = torch.cat(outputs, 1)

        return outputs

    def extract_features(self, x):
        feat_conv1 = []
        feat_conv2 = []
        feat_conv3 = []
        feat_conv4 = []
        for i in range(x.shape[1]):
            out = self.conv1(x[:, i, :])
            feat_conv1.append(out.unsqueeze(1))
            out = self.conv2(self.leakyReLU(out))
            feat_conv2.append(out.unsqueeze(1))
            out = self.conv3(self.leakyReLU(out))
            feat_conv3.append(out.unsqueeze(1))
            out = self.conv4(self.leakyReLU(out))
            feat_conv4.append(out.unsqueeze(1))
        feat_conv1 = torch.cat(feat_conv1, 1)
        feat_conv2 = torch.cat(feat_conv2, 1)
        feat_conv3 = torch.cat(feat_conv3, 1)
        feat_conv4 = torch.cat(feat_conv4, 1)

        return feat_conv1, feat_conv2, feat_conv3, feat_conv4
