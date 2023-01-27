import torch
import torch.nn as nn

from mygan.util import PixelNorm

class DiscBlock(nn.Module):
    '''
    Discriminator building block.
    '''
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, bias=False, padding_mode='reflect'),
            PixelNorm(),
            nn.Mish(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            # both source and target come in, thus double the amount of channels:
            nn.Conv2d(in_channels * 2, 64, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.Mish(inplace=True),
            DiscBlock(64, 128),
            DiscBlock(128, 256),
            DiscBlock(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'),
        )
        
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.model(x)
