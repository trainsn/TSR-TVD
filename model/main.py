# main file

import os
import argparse
import math
import pdb
import time

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

from generator import Generator
from discriminator import Discriminator
import sys
sys.path.append("../datasets")
from trainDataset import *
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
    parser.add_argument("--save-dir", required=True, type=str,
                        help="dir of the output models")
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
    parser.add_argument("--gan-loss-weight" , type=float, default=1e-3,
                        help="weight of the adversarial loss")
    parser.add_argument("--volume-loss-weight", type=float, default=1,
                        help="weight of the volume loss (mse)")
    parser.add_argument("--feature-loss-weight", type=float, default=5e-2,
                        help="weight of the feature loss")
    parser.add_argument("--wo-ori-volume", action="store_true", default=False,
                        help="during training, without the original volume")

    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate (default: 1e-4)")
    parser.add_argument("--d-lr", type=float, default=4e-4,
                        help="learning rate of the discriminator (default 4e-4)")
    parser.add_argument("--beta1", type=float, default=0.0,
                        help="beta1 of Adam (default: 0.0)")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="beta2 of Adam (default: 0.999)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size for training")
    parser.add_argument("--training-step", type=int, default=3,
                        help="in the training phase, the number of intermediate volumes")
    parser.add_argument("--n-d", type=int, default=2,
                        help="number of D updates per iteration")
    parser.add_argument("--n-g", type=int, default=1,
                        help="number of G upadates per iteration")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="start epoch number (default: 0)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of epochs to train")

    parser.add_argument("--log-every", type=int, default=3,
                        help="log training status every given number of batches")
    parser.add_argument("--test-every", type=int, default=9,
                        help="test every given number of sub-epochs (default: 5")
    parser.add_argument("--check-every", type=int, default=30,
                        help="save checkpoint every given number of sub-epochs (default: 20)")

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
    train_dataset = TVDataset(
        root=args.root,
        sub_size=args.block_size,
        volume_list="volume_train_list.txt",
        max_k=args.training_step,
        train=True,
        transform=transform
    )
    test_dataset = TVDataset(
        root=args.root,
        sub_size=args.block_size,
        volume_list="volume_test_list.txt",
        max_k=args.training_step,
        train=False,
        transform=transform
    )

    kwargs = {"num_workers": 4, "pin_memory": True} if args.cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=True, **kwargs)

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

    if args.gan_loss != "none":
        d_model = Discriminator()
        d_model.apply(discriminator_weights_init)
        if args.sn:
            d_model = add_sn(d_model)
        if args.data_parallel and torch.cuda.device_count() > 1:
            d_model = nn.DataParallel(d_model)
        d_model.to(device)

    mse_loss = nn.MSELoss()
    adversarial_loss = nn.MSELoss()
    train_losses, test_losses = [], []
    d_losses, g_losses = [], []

    # optimizer
    g_optimizer = optim.Adam(g_model.parameters(), lr=args.lr,
                             betas=(args.beta1, args.beta2))
    if args.gan_loss != "none":
        d_optimizer = optim.Adam(d_model.parameters(), lr=args.d_lr,
                                 betas=(args.beta1, args.beta2))

    Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

    # load checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            g_model.load_state_dict(checkpoint["g_model_state_dict"])
            # g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
            if args.gan_loss != "none":
                d_model.load_state_dict(checkpoint["d_model_state_dict"])
                # d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
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
        if args.gan_loss != "none":
            d_model.train()
        train_loss = 0.
        for i, sample in enumerate(train_loader):
            # adversarial ground truths
            real_label = Variable(Tensor(sample["v_i"].shape[0], sample["v_i"].shape[1], 1, 1, 1, 1).fill_(1.0), requires_grad=False)
            fake_label = Variable(Tensor(sample["v_i"].shape[0], sample["v_i"].shape[1], 1, 1, 1, 1).fill_(0.0), requires_grad=False)

            v_f = sample["v_f"].to(device)
            v_b = sample["v_b"].to(device)
            v_i = sample["v_i"].to(device)
            g_optimizer.zero_grad()
            fake_volumes = g_model(v_f, v_b, args.training_step, args.wo_ori_volume)

            # adversarial loss
            # update discriminator
            if args.gan_loss != "none":
                avg_d_loss = 0.
                avg_d_loss_real = 0.
                avg_d_loss_fake = 0.
                for k in range(args.n_d):
                    d_optimizer.zero_grad()
                    decisions = d_model(v_i)
                    d_loss_real = adversarial_loss(decisions, real_label)
                    fake_decisions = d_model(fake_volumes.detach())

                    d_loss_fake = adversarial_loss(fake_decisions, fake_label)
                    d_loss = d_loss_real + d_loss_fake
                    d_loss.backward()
                    avg_d_loss += d_loss.item() / args.n_d
                    avg_d_loss_real += d_loss_real / args.n_d
                    avg_d_loss_fake += d_loss_fake / args.n_d

                    d_optimizer.step()

            # update generator
            if args.gan_loss != "none":
                avg_g_loss = 0.
            avg_loss = 0.
            for k in range(args.n_g):
                loss = 0.
                g_optimizer.zero_grad()

                # adversarial loss
                if args.gan_loss != "none":
                    fake_decisions = d_model(fake_volumes)
                    g_loss = args.gan_loss_weight * adversarial_loss(fake_decisions, real_label)
                    loss += g_loss
                    avg_g_loss += g_loss.item() / args.n_g

                # volume loss
                if args.volume_loss:
                    volume_loss = args.volume_loss_weight * mse_loss(v_i, fake_volumes)
                    loss += volume_loss

                # feature loss
                if args.feature_loss:
                    feat_real = d_model.extract_features(v_i)
                    feat_fake = d_model.extract_features(fake_volumes)
                    for m in range(len(feat_real)):
                        loss += args.feature_loss_weight / len(feat_real) * mse_loss(feat_real[m], feat_fake[m])

                avg_loss += loss / args.n_g
                loss.backward()
                g_optimizer.step()

            train_loss += avg_loss

            # log training status
            subEpoch = (i + 1) // args.log_every
            if (i+1) % args.log_every == 0:
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, (i+1) * args.batch_size, len(train_loader.dataset), 100. * (i+1) / len(train_loader),
                    avg_loss
                ))
                if args.gan_loss != "none":
                    print("DLossReal: {:.6f} DLossFake: {:.6f} DLoss: {:.6f}, GLoss: {:.6f}".format(
                        avg_d_loss_real, avg_d_loss_fake, avg_d_loss, avg_g_loss
                    ))
                    d_losses.append(avg_d_loss)
                    g_losses.append(avg_g_loss)
                train_losses.append(avg_loss)
                print("====> SubEpoch: {} Average loss: {:.4f}".format(
                    subEpoch, train_loss / args.log_every
                ))
                train_loss = 0.

            # testing...
            if (i + 1) % args.test_every == 0:
                g_model.eval()
                if args.gan_loss != "none":
                    d_model.eval()
                test_loss = 0.
                with torch.no_grad():
                    for i, sample in enumerate(test_loader):
                        v_f = sample["v_f"].to(device)
                        v_b = sample["v_b"].to(device)
                        v_i = sample["v_i"].to(device)
                        fake_volumes = g_model(v_f, v_b, args.training_step, args.wo_ori_volume)
                        test_loss += args.volume_loss_weight * mse_loss(v_i, fake_volumes).item()

                test_losses.append(test_loss * args.batch_size / len(test_loader.dataset))
                print("====> SubEpoch: {} Test set loss {:4f} Time {}".format(
                    subEpoch, test_losses[-1], time.asctime( time.localtime(time.time()) )
                ))

            # saving...
            if (i+1) % args.check_every == 0:
                print("=> saving checkpoint at epoch {}".format(epoch))
                if args.gan_loss != "none":
                    torch.save({"epoch": epoch + 1,
                                "g_model_state_dict": g_model.state_dict(),
                                "g_optimizer_state_dict":  g_optimizer.state_dict(),
                                "d_model_state_dict": d_model.state_dict(),
                                "d_optimizer_state_dict": d_optimizer.state_dict(),
                                "d_losses": d_losses,
                                "g_losses": g_losses,
                                "train_losses": train_losses,
                                "test_losses": test_losses},
                               os.path.join(args.save_dir, "model_" + str(epoch) + "_" + str(subEpoch) + "_" + "pth.tar")
                               )
                else:
                    torch.save({"epoch": epoch + 1,
                                "g_model_state_dict": g_model.state_dict(),
                                "g_optimizer_state_dict": g_optimizer.state_dict(),
                                "train_losses": train_losses,
                                "test_losses": test_losses},
                               os.path.join(args.save_dir, "model_" + str(epoch) + "_" + str(subEpoch) + "_" + "pth.tar")
                               )
                torch.save(g_model.state_dict(),
                           os.path.join(args.save_dir, "model_" + str(epoch) + "_" + str(subEpoch) + ".pth"))

if __name__  == "__main__":
    main(parse_args())