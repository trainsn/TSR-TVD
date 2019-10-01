# Discriminator architecture

import torch
import torch.nn as nn

from basicblock import ConvLayer

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # volume classification subnet
        self.conv1 = ConvLayer(in_channels=1, out_channels=64, kernel_size=4, stride=2)
        self.conv2 = ConvLayer(in_channels=64, out_channels=128, kernel_size=4, stride=2)
        self.conv3 = ConvLayer(in_channels=128, out_channels=256, kernel_size=4, stride=2)
        self.conv4 = ConvLayer(in_channels=256, out_channels=512, kernel_size=4, stride=2)
        self.conv5 = nn.Conv3d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0)

        self.leakyRuLU =  nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        outputs = []
        for i in range(x.shape[0]):
            out = self.leakyRuLU(self.conv1(x[i]))
            out = self.leakyRuLU(self.conv2(out))
            out = self.leakyRuLU(self.conv3(out))
            out = self.leakyRuLU(self.conv4(out))
            out = self.tanh(self.conv5(out))
            outputs.appendt(out)
        outputs = torch.tensor(outputs)

        return outputs

    def extract_features(self, x, total_step):
        feat_conv1 = []
        feat_conv2 = []
        feat_conv3 = []
        feat_conv4 = []
        for i in range(total_step):
            out = self.conv1(x[i])
            feat_conv1.append(out)
            out = self.conv2(self.leakyRuLU(out))
            feat_conv2.append(out)
            out = self.conv3(self.leakyReLU(out))
            feat_conv3.append(out)
            out = self.conv4(self.leakyRuLU(out))
            feat_conv4.append(out)
        feat_conv1 = torch.tensor(feat_conv1)
        feat_conv2 = torch.tensor(feat_conv2)
        feat_conv3 = torch.tensor(feat_conv3)
        feat_conv4 = torch.tensor(feat_conv4)

        return feat_conv1, feat_conv2, feat_conv3, feat_conv4
