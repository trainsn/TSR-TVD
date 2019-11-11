import os
import struct
import numpy as np

import torch
from torch.utils.data import Dataset
from trainDataset import *

class InferTVDataset(Dataset):
    def __init__(self, root, sub_size, max_k, volume_list="volume_test_list.txt", transform=None,
                 loader=volume_loader):
        f = open(os.path.join(root, "test_cropped", volume_list))
        line = f.readline()
        self.dataset_size = int(line)

        self.vs = []
        line = f.readline()
        while line:
            if line[-1] == '\n':
                line = line[:-1]
            self.vs.append(os.path.join("test_cropped", line))
            line = f.readline()

        self.root = root
        self.sub_size = sub_size
        self.max_k = max_k
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        vf_path = os.path.join(self.root, self.vs[index])
        v_f = self.loader(vf_path, self.sub_size, self.sub_size, self.sub_size)

        vb_path = os.path.join(self.root, self.vs[self.dataset_size + index])
        v_b = self.loader(vb_path, self.sub_size, self.sub_size, self.sub_size)

        if self.transform is not None:
            v_f = self.transform(v_f)
            v_b = self.transform(v_b)

        sample = {
            "vf_name": self.vs[index],
            "vb_name": self.self.vs[self.dataset_size + index],
            "v_f": v_f, "v_b": v_b}

        return sample




