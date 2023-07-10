import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# baseline
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.interpolate(x, [160, 160], mode='bilinear', align_corners=True)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)
        return x

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNet1(nn.Module):
    def __init__(self, number_channels=64, number_residual_blocks=5, dim=1):
        super(ResNet1, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.res_blocks = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks.append(ResidualBlock(number_channels, number_channels))
        self.conv2 = nn.Sequential(nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(number_channels, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = F.interpolate(x, [160, 160], mode='bilinear', align_corners=True)
        out = self.conv1(out)
        out = self.conv2(out)
        for layer in self.res_blocks:
            out = layer(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return out


class ResNet2(nn.Module):
    def __init__(self, number_channels=64, number_residual_blocks=4, upsampling_factor=4, dim=1):
        super(ResNet2, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(dim, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.res_blocks = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks.append(ResidualBlock(number_channels, number_channels))
        self.conv2 = nn.Sequential(
            nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.upsampling = nn.ModuleList()
        for k in range(int(np.rint(np.log2(upsampling_factor)))):
            self.upsampling.append(nn.ConvTranspose2d(number_channels, number_channels, kernel_size=2, padding=0, stride=2))
        self.conv3 = nn.Sequential(nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Conv2d(number_channels, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = F.interpolate(x, [20, 20], mode='bilinear', align_corners=True)
        out = self.conv1(x)
        for layer in self.upsampling:
            out = layer(out)
        out = self.conv2(out)
        for layer in self.res_blocks:
            out = layer(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.unsqueeze(1)
        return out

class ESRGAN_Discriminator(nn.Module):
    def __init__(self):
        super(ESRGAN_Discriminator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv9 = nn.Conv2d(128, 1, 1, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv9(x)
        return torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)
