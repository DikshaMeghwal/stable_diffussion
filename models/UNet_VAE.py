import torch
import torch.nn as nn

# design choices:
# - normalization: layer norm/group norm
# - where to put the normalization layer? before conv, before activation
# - understand the input and output size expectations esp for say ConvTranspose2D

# model_config = {
#     'down_channels': [1, 16, 32, 64],
#     'down_kernel': [3, 3, 3],
#     'down_padding': [1, 1, 1],
#     'down_stride': [1,1,1],
#     'enc_fc_channels': [576, 128, 2],
#     'up_kernel': [3, 2, 2],
#     'up_padding': [0, 0, 0],
#     'up_stride': [2, 2, 2]
# }

class UNet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        down_channels = config['down_channels']
        down_kernel = config['down_kernel']
        down_stride = config['down_stride']
        down_padding = config['down_padding']
        self.down_blocks = nn.ModuleList(
            [
                DownBlock(
                    in_channels=down_channels[idx], 
                    out_channels=down_channels[idx+1], 
                    kernel_size=down_kernel[idx],
                    stride=down_stride[idx],
                    padding=down_padding[idx]
                ) for idx in range(len(down_channels) - 1)
            ]
        )
        self.latent_dim = config['latent_dim']
        self.downsample_blocks = [nn.MaxPool2d(kernel_size=2, stride=2)] * len(down_channels)
        enc_fc_channels = config['enc_fc_channels']
        self.mean_encoder = nn.ModuleList(
            [ 
                nn.Sequential( 
                    nn.Linear(enc_fc_channels[i], enc_fc_channels[i+1]),
                )
                for i in range(len(enc_fc_channels) - 1)
            ]
        )
        self.variance_encoder = nn.ModuleList(
            [ 
                nn.Sequential( 
                    nn.Linear(enc_fc_channels[i], enc_fc_channels[i+1]),
                    # nn.LeakyReLU()
                )
                for i in range(len(enc_fc_channels) - 1)
            ]
        )
        self.decoder_fc = nn.ModuleList(
            [
                nn.Sequential( 
                    nn.Linear(enc_fc_channels[i], enc_fc_channels[i-1]),
                    nn.LeakyReLU()
                ) for i in reversed(range(1, len(enc_fc_channels))) 
            ]
        )
        up_kernel = config['up_kernel']
        up_stride = config['up_stride']
        up_padding = config['up_padding']
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(
                    in_channels=list(reversed(down_channels))[idx], 
                    out_channels=list(reversed(down_channels))[idx+1],
                    kernel_size=up_kernel[idx],
                    stride=up_stride[idx],
                    padding=up_padding[idx]
                ) for idx in (range(len(down_channels) - 1))
            ]
        )


    def forward(self, input_img, label=None):
        x = input_img

        for (layer, downsample_layer) in zip(self.down_blocks, self.downsample_blocks):
            x = layer(x)
            x = downsample_layer(x)

        out = x.reshape(x.shape[0], -1)
        for layer in self.mean_encoder: 
            out = layer(out)
        mean = out 

        out = x.reshape(x.shape[0], -1)
        for layer in self.variance_encoder:
            out = layer(out)
        variance = out

        out = self.reparametrize(mean, variance)

        for layer in self.decoder_fc:
            out = layer(out)

        x = out.reshape(x.shape)
        for layer in self.up_blocks:
            x = layer(x)

        # return mean, variance, x
        return {
                'mean': mean,
                'log_variance': variance,
                'image': x,
            }


    def sample(self, label=None, num_images=1, z=None):
        if z is None:
            z = torch.randn((num_images, self.latent_dim))
        if self.config['conditional']:
            assert label is not None, "Label cannot be none for conditional sampling"
            assert label.size(0) == num_images
        assert z.size(0) == num_images
        out = self.generate(z, label)
        return out


    def reparametrize(self, mean, variance):
        random_factor = torch.rand_like(variance)
        return mean + random_factor * variance


    def generate(self, sample):
        out = sample
        for layer in self.decoder_fc:
            out = layer(out)
        B = out.shape[0]
        C = self.config['down_channels'][-1]
        H = W = int((out.shape[1] / C)**0.5)
        out = out.reshape((B, C, H, W))
        for idx,layer in enumerate(self.up_blocks):
            B,C,H,W = out.shape
            (H, W) = (2*H, 2*W) if idx > 0 else (2*H + 1, 2*W + 1)
            out = layer(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU()

    def forward(self, input_img):
        conv_out = self.conv(input_img)
        norm_out = self.norm(conv_out)
        activation_out = self.activation(norm_out)
        return activation_out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.Tanh()
        self.convtranspose2d = nn.ConvTranspose2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )


    def forward(self, input_img):
        upsample_img = self.convtranspose2d(input_img)
        norm_out = self.norm(upsample_img)
        activation_out = self.activation(norm_out)
        return activation_out
