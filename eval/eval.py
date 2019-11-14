# model evaluation

import os
import argparse
import math
import pdb

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision import transforms

import sys
sys.path.append("../")
sys.path.append("../datasets")
sys.path.append("../model")
from generator import Generator
from discriminator import Discriminator
from inferDataset import *
import utils

def parse_args():
    parser = argparse.ArgumentParser(description="Deep Learning Model")

    parser.add_argument("--no-cuda", action="store_true" , default=False,
                        help="disable CUDA training")
    parser.add_argument("--data-parallel", action="store_true", default=False,
                        help="enable data parallelism")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")

    parser.add_argument("--root", required=True, type=str,
                        help="root of the dataset")
    parser.add_argument("--save-pred", required=True, type=str,
                        help="dir of predicted volumes")
    parser.add_argument("--resume", type=str, default="",
                        help="path to the latest checkpoint (default: none)")

    parser.add_argument("--sn", action="store_true", default=False,
                        help="enable spectral normalization")

    parser.add_argument("--gan-loss", type=str, default="none",
                        help="gan loss (default: none)")
    parser.add_argument("--volume-loss", action="store_true", default=False,
                        help ="enable volume loss")
    parser.add_argument("--feature-loss", action="store_true", default=False,
                        help="enable feature loss")

    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate (default: 1e-4)")
    parser.add_argument("--d-lr", type=float, default=4e-4,
                        help="learning rate of the discriminator (default 4e-4)")
    parser.add_argument("--beta1", type=float, default=0.0,
                        help="beta1 of Adam (default: 0.0)")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="beta2 of Adam (default: 0.999)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="batch size for training (default: 4)")
    parser.add_argument("--infering-step", type=int, default=3,
                        help="in the infering phase, the number of intermediate volumes")
    parser.add_argument("--n-d", type=int, default=2,
                        help="number of D updates per iteration")
    parser.add_argument("--n-g", type=int, default=1,
                        help="number of G upadates per iteration")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="start epoch number (default: 0)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs to train (default: 10)")

    parser.add_argument("--block-size", type=int, default=64,
                        help="the size of the sub-block")
    return parser.parse_args()

# the main function
def main(args):
    # log hyperparameter
    print(args)

    # select device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda: 0" if args.cuda else "cpu")

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # data loader
    transform = transforms.Compose([
        utils.Normalize(),
        utils.ToTensor()
    ])

    infer_dataset = InferTVDataset(
        root=args.root,
        sub_size=args.block_size,
        volume_list="volume_test_list.txt",
        max_k = args.infering_step,
        transform=transform
    )

    kwargs = {"num_workers": 4, "pin_memory": True} if args.cuda else {}
    infer_loader = DataLoader(infer_dataset, batch_size=args.batch_size,
                             shuffle=False, **kwargs)

    # model
    def generator_weights_init(m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def discriminator_weights_init(m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def add_sn(m):
        for name, c in m.named_children():
            m.add_module(name, add_sn(c))
        if isinstance(m, nn.Conv2d):
            return nn.utils.spectral_norm(m, eps=1e-4)
        else:
            return m

    g_model = Generator()
    g_model.apply(generator_weights_init)
    if args.data_parallel and torch.cuda.device_count() > 1:
        g_model = nn.DataParallel(g_model)
    g_model.to(device)

    mse_loss = nn.MSELoss()
    adversarial_loss = nn.MSELoss()
    train_losses, test_losses = [], []
    d_losses, g_losses = [], []

    # load checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            g_model.load_state_dict(checkpoint["g_model_state_dict"])
            # g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
            if args.gan_loss != "none":
                # d_model.load_state_dict(checkpoint["d_model_state_dict"])
                # d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
                d_losses = checkpoint["d_losses"]
                g_losses = checkpoint["g_losses"]
            train_losses = checkpoint["train_losses"]
            test_losses = checkpoint["test_losses"]
            print("=> load chekcpoint {} (epoch {})"
                  .format(args.resume, checkpoint["epoch"]))

    g_model.eval()
    inferRes = []
    zSize, ySize, xSize = 120, 720, 480
    for i in range(args.infering_step):
        inferRes.append(np.zeros((zSize, ySize, xSize)))
    inferScale = np.zeros((zSize, ySize, xSize))
    time_start = 0
    volume_type = ''

    with torch.no_grad():
        for i, sample in tqdm(enumerate(infer_loader)):
            v_f = sample["v_f"].to(device)
            v_b = sample["v_b"].to(device)
            fake_volumes = g_model(v_f, v_b, args.infering_step)
            volume_type, time_start, x_start, y_start, z_start = utils.Parse(sample["vf_name"][0])

            for j in range(fake_volumes.shape[1]):
                volume = fake_volumes[0, j, 0]
                min_value = -0.015  # -0.012058
                max_value = 1.01  # 1.009666
                mean = (min_value + max_value) / 2
                std = mean - min_value
                volume = volume.to("cpu").numpy() * std + mean

                inferRes[j][z_start:z_start+args.block_size,
                y_start:y_start+args.block_size, x_start:x_start+args.block_size] += volume
                if j == 0:
                    inferScale[z_start:z_start + args.block_size,
                    y_start:y_start + args.block_size, x_start:x_start + args.block_size] += 1
                # pdb.set_trace()

    for j in range(args.infering_step):
        inferRes[j] = inferRes[j] / inferScale
        inferRes[j] = inferRes[j].astype(np.float32)

        volume_name = volume_type + '_' + ("%04d" % (time_start+j+1)) + '.raw'
        inferRes[j].tofile(os.path.join(args.save_pred, volume_name))

if __name__ == "__main__":
    main(parse_args())





