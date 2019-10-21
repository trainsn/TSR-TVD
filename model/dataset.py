import os
import struct
import numpy as np
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
    def __init__(self, root, block_size, train=True, transform=None,
                 loader=volume_loader):
        volume_list = 'volume_list.txt'
        f = open(os.path.join(root, volume_list))
        line = int(f.readline())
        self.dataset_size = int(line)
        self.v_fs = []
        self.v_bs = []
        self.v_is = []
        line = f.readline()
        while line:
            step = int(line)
            v_i = []
            for i in range(step):
                line = f.readline()
                if line[-1] == '\n':
                    line = line[:-1]
                if i == 0:
                    self.v_fs.append(line)
                elif i == step-1:
                    self.v_bs.append(line)
                else:
                    v_i.append(line)
            self.v_is.append(v_i)
            line = f.readline()

        self.root = root
        self.train = train
        self.transform = transform
        self.loader = loader
        self.block_size = block_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        vf_path = os.path.join(self.root, self.v_fs[index])
        v_f = self.loader(vf_path, self.block_size, self.block_size, self.block_size)
        vb_path = os.path.join(self.root, self.v_bs[index])
        v_b = self.loader(vb_path, self.block_size, self.block_size, self.block_size)
        if self.transform is not None:
            v_f = self.transform(v_f)
            v_b = self.transform(v_b)

        vi_paths = self.v_is[index]
        vi_list = []
        for i, vi_subpath in enumerate(vi_paths):
            vi_path = os.path.join(self.root, vi_subpath)
            v_i = self.loader(vi_path, self.block_size, self.block_size, self.block_size)
            if self.transform is not None:
                v_i = self.transform(v_i)
            v_i = torch.unsqueeze(v_i, 0)
            vi_list.append(v_i)

        v_is = torch.cat(vi_list, 0)
        sample = {"v_f": v_f, "v_b": v_b, "v_i": v_is}

        return sample

class Normalize(object):
    def __call__(self, volume):
        min_value = -0.01
        max_value = 1.01
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
# path = 'D:\\OSU\\Grade1\\Research\\exavisData\\combustion\\Jet_0016-0020\\jet_0016\\jet_mixfrac_0016_x413_y397_z34.raw'
# volume = volume_loader(path, 64, 64, 64)
# print("{} {}".format(np.min(volume), np.max(volume)))