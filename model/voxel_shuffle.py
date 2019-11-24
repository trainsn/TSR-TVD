import torch
from torch.autograd import Variable

import pdb

def pixel_shuffle(input, upscale_factor):
    batch_size, c, h, w, l = input.size()
    rh, rw, rl = upscale_factor, upscale_factor, upscale_factor
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

    # input_size = list(input.size())
    # dimensionality = len(input_size) - 2
    #
    # input_size[1] //= (upscale_factor ** dimensionality)
    # output_size = [dim*upscale_factor for dim in input_size[2:]]
    #
    # input_view = input.contiguous().view(
    #     input_size[0], input_size[1],
    #     *([upscale_factor]*dimensionality), *(input_size[2:])
    # )
    #
    # indicies = list(range(2, 2 + 2*dimensionality))
    # indicies = indicies[1::2] + indicies[0::2]
    #
    # shuffle_out = input_view.permute(0, 1, *(indicies[::-1])).contiguous()
    # return shuffle_out.view(input_size[0], input_size[1], *output_size)

input = Variable(torch.Tensor(1, 8, 2, 2, 2))
for i in range(8):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                input[0, i, j, k, l] = j * 32 + k * 16 + l * 8 + i
# input = Variable(torch.Tensor(1, 4, 2, 2))
# for i in range(4):
#     for j in range(2):
#         for k in range(2):
#             input[0, i, j, k] = j * 8 + k * 4 + i
print(input)
output = pixel_shuffle(input, upscale_factor=2)
print(output)
pdb.set_trace()
print(output.size())

