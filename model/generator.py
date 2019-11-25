# Generator architecture

import torch
import torch.nn as nn

from basicblock import ConvLayer, UpsampleConvLayer, ForwardBlockGenerator, BackwardBlockGenerator, ConvLSTMCell

import pdb

class Generator(nn.Module):
    def __init__(self, upsample_mode):
        super(Generator, self).__init__()

        # self.for_res1 = ForwardBlockGenerator(in_channels=1, out_channels=16, kernel_size=5, stride=1, downsample_factor=2)
        # self.for_res2 = ForwardBlockGenerator(in_channels=16, out_channels=32, kernel_size=3, stride=1, downsample_factor=2)
        # self.for_res3 = ForwardBlockGenerator(in_channels=32, out_channels=64, kernel_size=3, stride=1, downsample_factor=2)
        # self.for_res4 = ForwardBlockGenerator(in_channels=64, out_channels=64, kernel_size=3, stride=1, downsample_factor=2)
        #
        # #temporal component: convolution LSTM
        # self.num_layers = 1
        # self.temporal_subnet = []
        # for i in range(self.num_layers):
        #     name = 'cell{}'.format(i)
        #     cell = ConvLSTMCell(input_channels=64, hidden_channels=64, kernel_size=3, stride=1)
        #     setattr(self, name, cell)
        #     self.temporal_subnet.append(cell)
        #
        # self.back_res1 = BackwardBlockGenerator(in_channels=64, out_channels=64, kernel_size=3, stride=1, upsample_factor=2)
        # self.back_res2 = BackwardBlockGenerator(in_channels=64, out_channels=32, kernel_size=3, stride=1, upsample_factor=2)
        # self.back_res3 = BackwardBlockGenerator(in_channels=32, out_channels=16, kernel_size=3, stride=1, upsample_factor=2)
        # self.back_res4 = BackwardBlockGenerator(in_channels=16, out_channels=1, kernel_size=5, stride=1, upsample_factor=2)

        # feature learning component
        self.for_res1 = ForwardBlockGenerator(in_channels=1, out_channels=16, kernel_size=5, stride=1,
                                              downsample_factor=2)
        self.for_res2 = ForwardBlockGenerator(in_channels=16, out_channels=32, kernel_size=3, stride=1,
                                              downsample_factor=2)
        self.for_res3 = ForwardBlockGenerator(in_channels=32, out_channels=64, kernel_size=3, stride=1,
                                              downsample_factor=2)
        self.for_res4 = ForwardBlockGenerator(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                                              downsample_factor=2)

        self.back_res1 = BackwardBlockGenerator(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                                                upsample_mode=upsample_mode, upsample_factor=2)
        self.back_res2 = BackwardBlockGenerator(in_channels=64, out_channels=32, kernel_size=3, stride=1,
                                                upsample_mode=upsample_mode, upsample_factor=2)
        self.back_res3 = BackwardBlockGenerator(in_channels=32, out_channels=16, kernel_size=3, stride=1,
                                                upsample_mode=upsample_mode, upsample_factor=2)
        self.back_res4 = BackwardBlockGenerator(in_channels=16, out_channels=1, kernel_size=5, stride=1,
                                                upsample_mode=upsample_mode, upsample_factor=2)

        self.relu = torch.nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x_f):
        x = x_f

        x = self.for_res1(x)
        x = self.for_res2(x)
        x = self.for_res3(x)
        x = self.for_res4(x)

        x = self.back_res1(x)
        x = self.back_res2(x)
        x = self.back_res3(x)
        x = self.back_res4(x)
        y = self.tanh(x)

        return y

        # feature learning component
        # x = self.for_res1(x)
        # x = self.for_res2(x)
        # x = self.for_res3(x)
        # x = self.for_res4(x)

        # upscaling component
        # x = self.back_res1(x)
        # x = self.back_res2(x)
        # x = self.back_res3(x)
        # x = self.back_res4(x)
        # x = self.tanh(x)

        # return x


    # def forward(self, x_f, x_b, total_step, wo_ori_volume):
    #     # forward prediction
    #     internal_state = []
    #     outputs_f = []
    #     x = x_f
    #     for step in range(total_step):
    #         # feature learning component
    #         x = self.for_res1(x)
    #         x = self.for_res2(x)
    #         x = self.for_res3(x)
    #         x = self.for_res4(x)
    #
    #         # temporal component
    #         for i in range(self.num_layers):
    #             # all cells are initialized in the first step
    #             name = 'cell{}'.format(i)
    #             if step == 0:
    #                 bsize, _, height, length, width = x.size()
    #                 (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden_channels=64,
    #                                                          shape=(height, length, width))
    #                 internal_state.append((h, c))
    #             # do forward
    #             (h, c) = internal_state[i]
    #             x, new_c = getattr(self, name)(x, h, c)
    #             internal_state[i] = (x, new_c)
    #
    #         # upscaling component
    #         x = self.back_res1(x)
    #         x = self.back_res2(x)
    #         x = self.back_res3(x)
    #         x = self.back_res4(x)
    #         x = self.tanh(x)
    #
    #         # save result
    #         outputs_f.append(x)
    #     return x
    #
    #
    #     # backward prediction
    #     internal_state = []
    #     outputs_b = []
    #     x = x_b
    #     for step in range(total_step):
    #         # feature learning component
    #         x = self.for_res1(x)
    #         x = self.for_res2(x)
    #         x = self.for_res3(x)
    #         x = self.for_res4(x)
    #
    #         # temporal component
    #         for i in range(self.num_layers):
    #             # all cells are initialized in the first step
    #             name = 'cell{}'.format(i)
    #             if step == 0:
    #                 bsize, _, height, length, width = x.size()
    #                 (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden_channels=64,
    #                                                          shape=(height, length, width))
    #                 internal_state.append((h, c))
    #             # do backward
    #             (h, c) = internal_state[i]
    #             x, new_c = getattr(self, name)(x, h, c)
    #             internal_state[i] = (x, new_c)
    #
    #         # upscaling component
    #         x = self.back_res1(x)
    #         x = self.back_res2(x)
    #         x = self.back_res3(x)
    #         x = self.back_res4(x)
    #         x = self.tanh(x)
    #
    #         # save result
    #         outputs_b.append(x)
    #
    #     # blend module
    #     outputs = []
    #     for step in range(total_step):
    #         if wo_ori_volume:
    #             outputs.append(0.5 * (outputs_f[step] + outputs_b[total_step - 1 - step]))
    #         else:
    #             w = (step + 1) / (total_step + 1)
    #             lerp = (1 - w) * x_f + w * x_b
    #             outputs.append(lerp + 0.5 * (outputs_f[step] + outputs_b[total_step - 1 - step]))
    #         # outputs.append(lerp)
    #         outputs[step] = outputs[step].unsqueeze(1)
    #     outputs = torch.cat(outputs, 1)
    #
    #     return outputs


