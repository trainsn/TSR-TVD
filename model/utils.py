import numpy as np
import torch

class Normalize(object):
    def __call__(self, volume):
        min_value = -0.015 # -0.012058
        max_value = 1.01 # 1.009666
        mean = (min_value + max_value) / 2
        std = mean - min_value

        volume = (volume.astype(np.float32) - mean) / std
        return volume

class ToTensor(object):
    def __call__(self, volume):
        volume = torch.from_numpy(volume)
        volume = torch.unsqueeze(volume, 0)
        return volume

def Parse(volume_name):
    volume_type = volume_name[:volume_name.find('_')]
    volume_name = volume_name[volume_name.find('_') + 1:]
    volume_type += '_' + volume_name[:volume_name.find('_')]

    volume_name = volume_name[volume_name.find('_') + 1:]
    timestep = int(volume_name[:volume_name.find('_')])

    volume_name = volume_name[volume_name.find('_') + 1:]
    x_start =  int(volume_name[1:volume_name.find('_')])

    volume_name = volume_name[volume_name.find('_') + 1:]
    y_start = int(volume_name[1:volume_name.find('_')])

    volume_name = volume_name[volume_name.find('_') + 1:]
    z_start = int(volume_name[1:volume_name.find('_')])

    return volume_type, timestep, x_start, y_start, z_start
