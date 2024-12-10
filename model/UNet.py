import torch
import torch.nn as nn

# design choices:
# - normalization: layer norm/group norm
# - where to put the normalization layer? before conv, before activation
# - understand the input and output size expectations esp for say ConvTranspose2D
# - 

class UNet(nn.Module):
    def __init__(self, down_channels = [64, 128], in_channel=3) -> None:
        super().__init__()
        down_channels = [in_channel] + down_channels
        self.down_blocks = nn.ModuleList(
            [DownBlock(down_channels[idx], down_channels[idx+1]) for idx in range(len(down_channels) - 1)]
        )
        self.downsample_blocks = [nn.MaxPool2d(kernel_size=2, stride=2)] * len(down_channels)
        self.mid_blocks = nn.ModuleList(
            [
                MidBlock(down_channels[-1], down_channels[-1] * 2),
                MidBlock(down_channels[-1] * 2, down_channels[-1])
            ]
        )
        self.up_blocks = nn.ModuleList(
            [UpBlock(down_channels[idx], down_channels[idx-1]) for idx in reversed(range(1, len(down_channels)))]
        )
        self.out_activation = nn.Sigmoid()


    def forward(self, input_img):
        x = input_img
        down_outs = []

        for (layer, downsample_layer) in zip(self.down_blocks, self.downsample_blocks):
            x = layer(x)
            down_outs.append(x)
            x = downsample_layer(x)

        for layer in self.mid_blocks:
            x = layer(x)

        for (layer,down_out) in zip(self.up_blocks, reversed(down_outs)):
            x = layer(x, down_out)

        return self.out_activation(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, input_img):
        conv_out = self.conv(input_img)
        norm_out = self.norm(conv_out)
        activation_out = self.activation(norm_out)
        return activation_out


class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, input_img):
        conv_out = self.conv2d(input_img)
        norm_out = self.norm(conv_out)
        activation_out = self.activation(norm_out)
        return activation_out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=2*in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.convtranspose2d = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=2, padding=1)


    def forward(self, input_img, input_skip_connection):
        upsample_img = self.convtranspose2d(input_img)
        stacked_img = torch.concat([upsample_img, input_skip_connection], dim=1)
        conv_out = self.conv2d(stacked_img)
        norm_out = self.norm(conv_out)
        activation_out = self.activation(norm_out)
        return activation_out

