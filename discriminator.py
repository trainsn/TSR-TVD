# Discriminator architecture

import torch
import torch.nn as nn

from basicblock import ConvLayer

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # volume classification subnet
        self.volume_subnet = nn.Sequential(
            ConvLayer(in_channels=1, out_channels=64, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            ConvLayer(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            ConvLayer(in_channels=128, out_channels=256, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            ConvLayer(in_channels=256, out_channels=512, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.volume_subnet(x)
        return out


