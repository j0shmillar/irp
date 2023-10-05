"""
`ResNet` implementation modified from "Hard-Constrained Deep Learning for Climate Downscaling" (Harder et al., 2022)
Code found here:
    https://github.com/paulaharder/constrained-downscaling/blob/main/models.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels: int, out_channels: int, stride: int = 1):
    """
    3x3 convolutional layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride value for the convolutional layer. Default is 1.

    Returns:
        nn.Conv2d: Convolutional layer with 3x3 kernel.
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class ResBlock(nn.Module):
    "Residual Block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int, optional): Stride value for conv layers. Default is 1.
        """
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += res
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    "ResNet for 10x image SR (prog-upsampling)"

    def __init__(self, number_channels: int = 64, number_residual_blocks: int = 4, dim: int = 1):
        """
        Args:
            number_channels (int, optional): Number of channels ... Default is 64.
            number_residual_blocks (int, optional): Number of residual blocks. Default is 4.
            dim (int, optional): Number of input/output channels. Default is 1.
        """
        super(ResNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(nn.Conv2d(dim, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.res_blocks_2 = nn.ModuleList(ResBlock(number_channels, number_channels) for _ in range(number_residual_blocks))
        self.conv2 = nn.Sequential(nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.upsampling_1 = nn.ConvTranspose2d(number_channels, number_channels, kernel_size=2, padding=0, stride=2)
        self.res_blocks = nn.ModuleList(ResBlock(number_channels, number_channels) for _ in range(number_residual_blocks))
        self.upsampling_2 = nn.ConvTranspose2d(number_channels, number_channels, kernel_size=5, padding=0, stride=5)
        self.conv3 = nn.Sequential(nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Conv2d(number_channels, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 0, ...]
        out = self.conv1(x)
        out = self.upsampling_1(out)
        for layer in self.res_blocks_2:
            out = layer(out)
        out = self.upsampling_2(out)
        out = self.conv2(out)
        for layer in self.res_blocks:
            out = layer(out)
        out = self.conv3(out)
        out = self.relu(self.conv4(out))
        out = out.unsqueeze(1)
        return out


class ResNet_pre(nn.Module):
    "ResNet for 10x image SR (modified to pre-upsampling for comparison)"

    def __init__(self, number_channels: int = 64, number_residual_blocks: int = 5, upsampling_factor: int = 10, dim: int = 1):
        """
        Args:
            number_channels (int, optional): Number of channels ... Default is 64.
            number_residual_blocks (int, optional): Number of residual blocks. Default is 5.
            upsampling_factor (int, optional): Upsampling factor. Default is 10.
            dim (int, optional): Number of input/output dimensions. Default is 1.
        """
        super(ResNet_pre, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.res_blocks = nn.ModuleList([ResBlock(number_channels, number_channels) for _ in range(number_residual_blocks)])
        self.conv2 = nn.Sequential(nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Conv2d(number_channels, dim, kernel_size=1, stride=1, padding=0)
        self.upsampling_factor = upsampling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 0, ...]
        n_dims = [x.shape[2]*self.upsampling_factor, x.shape[3]*self.upsampling_factor]
        out = F.interpolate(x, n_dims, mode='bicubic', align_corners=True)
        out = self.conv1(out)
        out = self.conv2(out)
        for layer in self.res_blocks:
            out = layer(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return out


class Discriminator(nn.Module):
    "Discriminator model for adversarial training."

    def __init__(self, in_channels: int = 1, kernel_size: int = 3):
        """
        Args:
            in_channels (int, optional): Number of input channels. Default is 1.
            kernel_size (int, optional): Kernel size for convolutional layers. Default is 3.
        """
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(64, 128, kernel_size, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv6 = nn.Conv2d(128, 1, 1, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 0, ...]
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = torch.sigmoid(F.avg_pool2d(out, out.size()[2:])).view(out.size()[0], -1)
        return out
