# Generator architecture

import torch
import torch.nn as nn

from basicblock import ConvLayer, UpsampleConvLayer
from basicblock import ForwardBlockGenerator, BackwardBlockGenerator, ConvLSTMCell

import pdb

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        #feature learning component
        self.conv1 = ConvLayer(in_channels=1, out_channels=16, kernel_size=5, stride=2)
        self.in1 = nn.InstanceNorm3d(16, affine=True)
        self.conv2 = ConvLayer(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm3d(32, affine=True)
        self.conv3 = ConvLayer(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm3d(64, affine=True)
        self.conv4 = ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=2)
        self.in4 = nn.InstanceNorm3d(64, affine=True)

        #temporal component: convolution LSTM
        self.num_layers = 1
        self.temporal_subnet = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(input_channels=64, hidden_channels=64, kernel_size=3, stride=1)
            setattr(self, name, cell)
            self.temporal_subnet.append(cell)

        self.deconv1 = UpsampleConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm3d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(in_channels=64, out_channels=32, kernel_size=3, stride=1, upsample=2)
        self.in6 = nn.InstanceNorm3d(32, affine=True)
        self.deconv3 = UpsampleConvLayer(in_channels=32, out_channels=16, kernel_size=3, stride=1, upsample=2)
        self.in7 = nn.InstanceNorm3d(16, affine=True)
        self.deconv4 = UpsampleConvLayer(in_channels=16, out_channels=1, kernel_size=5, stride=1, upsample=2)

        self.relu = torch.nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x_f, x_b, total_step, wo_ori_volume):
        # forward prediction
        internal_state = []
        outputs_f = []
        x = x_f
        for step in range(total_step):
            # feature learning component
            x = self.relu(self.in1(self.conv1(x)))
            x = self.relu(self.in2(self.conv2(x)))
            x = self.relu(self.in3(self.conv3(x)))
            x = self.relu(self.in4(self.conv4(x)))

            # temporal component
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, length, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden_channels=64,
                                                             shape=(height, length, width))
                    internal_state.append((h, c))
                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)

            # upscaling component
            x = self.relu(self.in5(self.deconv1(x)))
            x = self.relu(self.in6(self.deconv2(x)))
            x = self.relu(self.in7(self.deconv3(x)))
            x = self.tanh(self.deconv4(x))

            # save result
            outputs_f.append(x)


        # backward prediction
        internal_state = []
        outputs_b = []
        x = x_b
        for step in range(total_step):
            # feature learning component
            x = self.relu(self.in1(self.conv1(x)))
            x = self.relu(self.in2(self.conv2(x)))
            x = self.relu(self.in3(self.conv3(x)))
            x = self.relu(self.in4(self.conv4(x)))

            # temporal component
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, length, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden_channels=64,
                                                             shape=(height, length, width))
                    internal_state.append((h, c))
                # do backward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)

            # upscaling component
            x = self.relu(self.in5(self.deconv1(x)))
            x = self.relu(self.in6(self.deconv2(x)))
            x = self.relu(self.in7(self.deconv3(x)))
            x = self.tanh(self.deconv4(x))

            # save result
            outputs_b.append(x)

        # blend module
        outputs = []
        for step in range(total_step):
            if wo_ori_volume:
                outputs.append(0.5 * (outputs_f[step] + outputs_b[total_step - 1 - step]))
            else:
                w = (step + 1) / (total_step + 1)
                lerp = (1 - w) * x_f + w * x_b
                outputs.append(lerp + 0.5 * (outputs_f[step] + outputs_b[total_step - 1 - step]))
            # outputs.append(lerp)
            outputs[step] = outputs[step].unsqueeze(1)
        outputs = torch.cat(outputs, 1)

        return outputs


