# main file

import os
import argparse
import math

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

from generator import Generator
from discriminator import Discriminator

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
    parser.add_argument("--resume", type=str, default="",
                        help="path to the latest checkpoint (default: none)")

    parser.add_argument("--sn", action="store_true", default=False,
                        help="enable spectral normalization")

    parser.add_argument("--gan_loss_weight" , type=float, default=1e-3,
                        help="weight of the adversarial loss")
    parser.add_argument("--volume_loss_weight", type=float, default=1,
                        help="weight of the volume loss (mse)")
    parser.add_argument("--feature_loss_weight", type=float, default=5e-2,
                        help="weight of the feature loss")

    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate (default: 1e-4)")
    parser.add_argument("--d-lr", type=float, default=4e-4,
                        help="learning rate of the discriminator (default 4e-4)")
    parser.add_argument("--beta1", type=float, default=0.0,
                        help="beta1 of Adam (default: 0.0)")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="beta2 of Adam (default: 0.999)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="batch size for training (default: 1)")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="start epoch number (default: 0)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of epochs to train (default: 10)")

    parser.add_argument("--log-every", type=int, default=10,
                        help="log training status every given number of batches (default: 10)")
    parser.add_argument("--check-every", type=int, default=20,
                        help="save checkpoint every given number of epochss (default: 20)")
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
    train_dataset = None
    test_dataset = None

    kwargs = {"num_workers": 4, "pin_memory": True} if args.cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle,True, **kwargs)

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

    d_model = Discriminator()
    d_model.apply(discriminator_weights_init)
    if args.sn:
        d_model = add_sn(d_model)
    if args.data_parallel and torch.cuda.device_count() > 1:
        d_model = nn.DataParallel(d_model)
    d_model.to(device)

    mse_criterion = nn.MSELoss()
    train_losses, test_losses = [], []
    d_losses, g_losses = [], []

    # optimizer
    g_optimizer = optim.Adam(g_model.parameter(), lr=args.lr,
                             betas=(args.beta1, args.beta2))
    d_optimizer = optim.Adam(d_model.parameter(), lr=args.lr,
                             betas=(args.beta1, args.beta2))

    # load checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            g_model.load_state_dict(checkpoint["g_model_state_dict"])
            g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
            d_model.load_state_dict(checkpoint["d_model_state_dict"])
            d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
            d_losses = checkpoint["d_losses"]
            g_losses = checkpoint["g_losses"]
            train_losses = checkpoint["train_losses"]
            test_losses = checkpoint["test_losses"]
            print("=> load chekcpoint {} (epoch {})"
                  .format(args.resume, checkpoint["epoch"]))

    # main loop
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        # training..
        g_model.train()
        d_model.train()
        train_loss = 0.
        
