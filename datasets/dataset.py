import os
import struct
import numpy as np
import random
import pdb

import torch
from torch.utils.data import Dataset

def volume_loader(path, zSize, ySize, xSize):
    f = open(path, 'rb')
    volume = np.zeros((zSize, ySize, xSize))
    for i in range(zSize):
        for j in range(ySize):
            for k in range(xSize):
                data = f.read(4)
                elem = struct.unpack("f", data)[0]
                volume[i][j][k] = elem
    f.close()
    return volume

class TVDataset(Dataset):
    def __init__(self, root, sub_size, max_k, volume_list="volume_train_list.txt", train=True, transform=None,
                 loader=volume_loader):
        if train:
            f = open(os.path.join(root, "train_cropped", volume_list))
        else
            f = open(os.path.join(root, "test_cropped", volume_list))
        line = f.readline()
        self.dataSize = int(line)
        line = f.readline()
        self.timeRange = int(line)

        self.vs = []
        line = f.readline()
        while line:
            if line[-1] == '\n':
                line = line[:-1]
            self.vs.append(line)
            line = f.readline()

        self.dataSize = self.dataSize * self.timeRange
        self.root = root
        self.sub_size = sub_size
        self.max_k = max_k
        self.train = train
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        vf_path = os.path.join(self.root, self.vs[index*(self.max_k+2)])
        v_f = self.loader(vf_path, self.sub_size, self.sub_size, self.sub_size)

        vb_path = os.path.join(self.root, self.vs[index*(self.max_k+2) + self.max_k + 1])
        v_b = self.loader(vb_path, self.sub_size, self.sub_size, self.sub_size)
        if self.transform is not None:
            v_f = self.transform(v_f)
            v_b = self.transform(v_b)

        vi_list = []
        for idx in range(index*(self.max_k+2) + 1, index*(self.max_k+2) + self.max_k + 1):
            vi_path = os.path.join(self.root, self.vs[idx])
            v_i = self.loader(vi_path, self.sub_size, self.sub_size, self.sub_size)
            if self.transform is not None:
                v_i = self.transform(v_i)
            v_i = torch.unsqueeze(v_i, 0)
            vi_list.append(v_i)

        v_is = torch.cat(vi_list, 0)
        sample = {"v_f": v_f, "v_b": v_b, "v_i": v_is}

        return sample

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

# volume_loader verification
# path = 'D:\\OSU\\Grade1\\Research\\TSR-TVD\\exavisData\\combustion\\jet_0016\\jet_mixfrac_0016.dat'
# volume = volume_loader(path, 120, 720, 480, 64)
# print("{} {}".format(np.min(volume), np.max(volume)))
# volume.tofile("sub_volume.raw")