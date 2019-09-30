# Generator architecture

import torch
import torch.nn as nn

from basicblock import ForwardBlockGenerator, BackwardBlockGenerator, ConvLSTMCell

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__(self)

        self.feature_learning_subnet = nn.Sequential(
            ForwardBlockGenerator(in_channels=1, out_channels=16, kernel_size=5, stride=1, downsample_factor=2),
            ForwardBlockGenerator(in_channels=16, out_channels=32, kernel_size=3, stride=1, downsample_factor=2),
            ForwardBlockGenerator(in_channels=32, out_channels=64, kernel_size=3, stride=1, downsample_factor=2),
            ForwardBlockGenerator(in_channels=64, out_channels=64, kernel_size=3, stride=1, downsample_factor=2)
        )

        #temporal component: convolution LSTM
        self.num_layers = 1
        self.temporal_subnet = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(input_channels=64, hidden_channels=64, kernel_size=3, stride=1)
            setattr(self, name, cell)
            self.temporal_subnet.append(cell)

        self.upscale_subnet = nn.Sequential(
            BackwardBlockGenerator(in_channels=64, out_channels=64, kernel_size=3, stride=1, upsample_factor=2),
            BackwardBlockGenerator(in_channels=64, out_channels=32, kernel_size=3, stride=1, upsample_factor=2),
            BackwardBlockGenerator(in_channels=32, out_channels=16, kernel_size=3, stride=1, upsample_factor=2),
            BackwardBlockGenerator(in_channels=16, out_channels=1, kernel_size=3, stride=1, upsample_factor=2),
            nn.Tanh()
        )

    def forward(self, x_f, x_b, total_step):
        # forward prediction
        internal_state = []
        outputs_f = []
        x = x_f
        for step in range(total_step):
            # feature learning component
            x = self.feature_learning_subnet(x)

            # temporal component
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden_channels=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))
                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)

            # upscaling component
            x = self.upscale_subnet(x)

            # save result
            outputs_f.append(x)


        # backward prediction
        internal_state = []
        outputs_b = []
        x = x_b
        for step in range(total_step):
            # feature learning component
            x = self.feature_learning_subnet(x)

            # temporal component
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden_channels=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))
                # do backward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)

            # upscaling component
            x = self.upscale_subnet(x)

            # save result
            outputs_b.append(x)

        # blend module
        outputs = []
        for step in range(total_step):
            w = (step+1) / (total_step+1)
            lerp = w * x_f + (1-w) * x_b
            outputs.append(0.5 * lerp + 0.5 * (outputs_f[step] + outputs_b[total_step-1-step]))

        return outputs
