import torch
import torch.nn as nn
from torch.autograd import Variable

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
        return (Variable(torch.zeros(batch_size, hidden_channels, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden_channels, shape[0], shape[1])).cuda())




