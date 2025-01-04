import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms
from argparse import ArgumentParser
from dataset.mnist import MNIST
from models.UNet_AE import UNet
from tqdm import tqdm
import numpy as np

# prepare dataset
transformations = torchvision.transforms.Compose([
    transforms.ToTensor()
])

train_dataset = MNIST(data_dir="data/mnist", split="train", transforms=transformations)
test_dataset = MNIST(data_dir="data/mnist", split="test", transforms=transformations)

# load data
num_workers = 0
batch_size = 32
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# initialize model
model = UNet(in_channel=1)
optimizer = Adam(model.parameters(), lr=1e-5)
# loss fn
criterion = nn.MSELoss()
losses = []

prog_bar = tqdm(train_dataloader)
prog_bar.set_description("Training Auto-Encoder...")

model.train()
num_epochs = 100
for _ in range(num_epochs):
    for img in prog_bar:
        optimizer.zero_grad()
        recon_img = model(img)
        loss = criterion(recon_img, img)
        loss.backward()
        losses.append(loss.detach())
        optimizer.step()
        prog_bar.set_description(f"Train loss:{np.mean(losses)}")

