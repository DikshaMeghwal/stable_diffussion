import os
import cv2
import torch
import torch.nn as nn
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from torchvision.utils import make_grid, save_image
import torchvision.transforms as transforms
from dataset.mnist import MNIST
from models.UNet_VAE import UNet
from tqdm import tqdm
import numpy as np


parser = ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--batch_size", metavar='b', type=int, default=32)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--save_image_epochs", type=int, default=1, help="num of epochs after which to save image during training")
parser.add_argument("--generate_mode", action='store_true', help='if you wish to generate images using the model')
parser.add_argument("--checkpoint_path", type=str, help='model checkpoint location')

args = parser.parse_args()
# prepare dataset
transformations = torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5))
])

if not args.generate_mode:
    train_dataset = MNIST(data_dir="data/mnist", split="train", transforms=transformations)
    test_dataset = MNIST(data_dir="data/mnist", split="test", transforms=transformations)

    # load data

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # model config
    model_config = {
        'down_channels': [1, 16, 32, 64],
        'down_kernel': [3, 3, 3],
        'down_padding': [1, 1, 1],
        'down_stride': [1,1,1],
        'enc_fc_channels': [576, 128, 2],
        'up_kernel': [3, 2, 2],
        'up_padding': [0, 0, 0],
        'up_stride': [2, 2, 2],
        'latent_dim': 2
    }

    # initialize model
    model = UNet(model_config)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose=True)
    # loss fn
    kl_loss_weight = 1e-4
    recon_loss = nn.MSELoss()
    kl_divergence_loss = lambda mean, log_variance: torch.mean(0.5 * torch.sum(torch.exp(log_variance) + mean ** 2 - 1 - log_variance, dim=-1))
    losses = []

    num_epochs = args.epochs

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}...")
        model.train()
        train_prog_bar = tqdm(train_dataloader)
        train_prog_bar.set_description("Training VAE...")

        for img in train_prog_bar:
            optimizer.zero_grad()
            output = model(img)
            mean, log_var, recon_img = output['mean'], output['log_variance'], output['image']
            # print(f"mean:{mean}, variance:{log_var}")

            loss = recon_loss(recon_img, img) + kl_loss_weight * kl_divergence_loss(mean, log_var)

            loss.backward()
            losses.append(loss.detach())
            breakpoint()
            optimizer.step()
            scheduler.step(loss)
            train_prog_bar.set_description(f"Train loss [{epoch}/{num_epochs}]:{np.mean(losses)} recon loss:{recon_loss(recon_img, img)} kl loss:{kl_divergence_loss(mean, log_var)}")

        if epoch % args.save_image_epochs == 0:
            os.makedirs(args.exp_name, exist_ok=True)
            org_img = make_grid(img)
            recon_img = make_grid(recon_img)
            save_image(org_img, f"{args.exp_name}/{epoch}_org.png")
            save_image(recon_img, f"{args.exp_name}/{epoch}_recon.png")

        with torch.no_grad():
            model.eval()
            val_prog_bar = tqdm(test_dataloader)
            val_prog_bar.set_description("Validating VAE...")

            for img in val_prog_bar:
                output = model(img)
                mean, log_var, recon_img = output['mean'], output['log_variance'], output['image']

                loss = recon_loss(recon_img, img) + kl_loss_weight * kl_divergence_loss(mean, log_var)
                losses.append(loss.detach())
                val_prog_bar.set_description(f"Validation loss [{epoch}/{num_epochs}]:{np.mean(losses)}")
        torch.save(model, f"{args.exp_name}/MNIST_VAE_{epoch}.pth")
        print("Finished 1 epoch")
else:
    assert args.checkpoint_path is not None
    # model config
    model_config = {
        'down_channels': [1, 16, 32, 64],
        'down_kernel': [3, 3, 3],
        'down_padding': [1, 1, 1],
        'down_stride': [1, 1, 1],
        'enc_fc_channels': [576, 128, 2],
        'up_kernel': [3, 2, 2],
        'up_padding': [0, 0, 0],
        'up_stride': [2, 2, 2]
    }
    # initialize model
    model = UNet(model_config)
    model.load_state_dict(torch.load(args.checkpoint_path).state_dict())
    grid = torch.cartesian_prod(torch.arange(-1, 1, 0.1), torch.arange(-1, 1, 0.1))
    recon_grid = model.generate(grid)
    recon_grid_img = make_grid(recon_grid, nrow=20)
    save_image(recon_grid_img,  f"{args.exp_name}/MNIST_VAE_manifold.png")
