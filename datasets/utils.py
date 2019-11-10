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