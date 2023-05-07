# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import random
import shutil
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

# from compressai.datasets import ImageFolder
# from compressai.losses import RateDistortionLoss
# from compressai.optimizers import net_aux_optimizer
# from compressai.zoo import image_models, bmshj2018_factorized, mbt2018_mean, cheng2020_attn, mbt2018
from image import ImageFolder
from rate_distortion import RateDistortionLoss
from net_aux import net_aux_optimizer
from models import image_models, bmshj2018_factorized, mbt2018_mean, cheng2020_attn, mbt2018


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, #aux_optimizer,#
    epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    train_iter = tqdm(train_dataloader)

    loss_meter = AverageMeter()
    mse_loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    bpp_loss_meter = AverageMeter()

    for i, d in enumerate(train_iter):
        #break
        d = d.to(device)

        optimizer.zero_grad()
        #aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        # aux_loss = model.aux_loss()
        # aux_loss.backward()
        # aux_optimizer.step()

        loss_meter.update(out_criterion["loss"].item())
        mse_loss_meter.update(out_criterion["mse_loss"].item())
        psnr_meter.update(out_criterion["psnr"])
        bpp_loss_meter.update(out_criterion["bpp_loss"].item())

        train_iter.set_description(
            f"epoch {epoch}: ["
            f"{i*len(d)}/{len(train_dataloader.dataset)}"
            f" ({100. * i / len(train_dataloader):.0f}%)]"
            #f'L: {out_criterion["loss"].item():.3f} ({loss_meter.avg:.3f})|'
            f'M: {out_criterion["mse_loss"].item():.4f} ({mse_loss_meter.avg:.4f})|'
            f'P: {out_criterion["psnr"]:.2f} ({psnr_meter.avg:.2f})|'
            f'B: {out_criterion["bpp_loss"].item():.3f} ({bpp_loss_meter.avg:.3f})|'
        )


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            psnr.update(out_criterion["psnr"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tPSNR: {psnr.avg:.2f} |"
        f"\tBpp loss: {bpp_loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.4f} |"
    )

    return loss.avg


def save_checkpoint(state, is_best, lmbda, model_name, savedir):
    ckpt_path = savedir + f'{model_name}_{lmbda}_ckpt.pth'
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copyfile(ckpt_path, savedir + f'{model_name}_{lmbda}_best.pth')


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        #default="mbt2018-mean",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=1500,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        #default=1e-6,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--savedir", type=str, default='backup/', help="Path to save")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # train_transforms = transforms.Compose(
    #     [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    # )
    train_transforms = transforms.Compose(
        [transforms.RandomResizedCrop(size=256, scale=(1.0, 1.0)), transforms.ToTensor()]
    )

    # test_transforms = transforms.Compose(
    #     [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    # )
    test_transforms = transforms.ToTensor()

    # transform = T.Compose([
    #     T.Resize(args.patch_size),
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.485, 0.456, 0.406],
    #                 std=[0.229, 0.224, 0.225])])

    train_dataset = ImageFolder("/home/weiluo6/CompressAI/compressai/datasets/" + args.dataset, transform=train_transforms)
    #test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)
    test_dataset = ImageFolder("/home/weiluo6/CompressAI/compressai/datasets/Kodak-Lossless-True-Color-Image-Suite/PhotoCD_PCD0992", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = image_models[args.model](quality=4)
    net = net.to(device)

    # if args.cuda and torch.cuda.device_count() > 1:
    #     net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    # parameters = net.optim_parameters()
    # optimizer = torch.optim.Adam([{'params': parameters}], lr=1e-6#, weight_decay=5e-4
    # )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    # TODO: BASELINE
    factorizedprior_model = bmshj2018_factorized(quality=4, metric='mse', pretrained=True, progress=True)
    net.load_state_dict(factorizedprior_model.state_dict())

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        #lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        #comment if want to skip train
        # train_one_epoch(
        #     net,
        #     criterion,
        #     train_dataloader,
        #     optimizer,
        #     #aux_optimizer,
        #     epoch,
        #     args.clip_max_norm,
        # )
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    #"aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                args.lmbda,
                args.model,
                args.savedir,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
