import torch
import torch.nn as nn

from mygan import config
from mygan.util import PixelNorm


class GenBlock(nn.Module):
    '''
    Generator building block.
    '''
    def __init__(self, in_channels, out_channels, down=True, act='relu', use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode='reflect') if down else
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            PixelNorm(),
            nn.ReLU(inplace=True) if act=='relu' else nn.Mish(inplace=True),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.conv(x)
        return x #self.dropout(x) if self.use_dropout else x
        
class Generator(nn.Module):
    def __init__(self, in_channels=3, n_features=64):
        super().__init__()
        self.n_features = n_features
        
        # No normalizing here.
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, n_features, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.Mish(inplace=True)
        )
        self.down1 = GenBlock(n_features, n_features*2, act='mish')
        self.down2 = GenBlock(n_features*2, n_features*4, act='mish')
        self.down3 = GenBlock(n_features*4, n_features*8, act='mish')
        self.down4 = GenBlock(n_features*8, n_features*8, act='mish')
        self.down5 = GenBlock(n_features*8, n_features*8, act='mish')
        self.down6 = GenBlock(n_features*8, n_features*8, act='mish')
        
        self.embedding = nn.Linear(config.colormap_size * 3, self.n_features)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(n_features*8, n_features*8, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            #nn.Mish(inplace=True),
        )
        
        self.up1 = GenBlock(n_features*9, n_features*8, down=False, use_dropout=True)
        self.up2 = GenBlock(n_features*16, n_features*8, down=False, use_dropout=True)
        self.up3 = GenBlock(n_features*16, n_features*8, down=False, use_dropout=True)
        self.up4 = GenBlock(n_features*16, n_features*8, down=False)
        self.up5 = GenBlock(n_features*16, n_features*4, down=False)
        self.up6 = GenBlock(n_features*8, n_features*2, down=False)
        self.up7 = GenBlock(n_features*4, n_features, down=False)
    
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(n_features*2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x, colorgram=None):
        if colorgram == None:
            rgb = torch.rand(config.colormap_size * 3).to(config.device).repeat(x.shape[0],1)
        else:
            rgb = colorgram
        
        # Normalize colors:
        rgb = rgb - 0.5
        
        initial = self.initial_down(x)
        conv1 = self.down1(initial)
        conv2 = self.down2(conv1)
        conv3 = self.down3(conv2)
        conv4 = self.down4(conv3)
        conv5 = self.down5(conv4)
        conv6 = self.down6(conv5)
        
        mid = self.bottleneck(conv6)
        embed = self.embedding(rgb)
        latent = torch.cat([mid, embed.unflatten(1, torch.Size(([self.n_features, 1, 1])))], dim=1)
        
        upconv1 = self.up1(latent)
        upconv2 = self.up2(torch.cat([upconv1, conv6], dim=1))
        upconv3 = self.up3(torch.cat([upconv2, conv5], dim=1))
        upconv4 = self.up4(torch.cat([upconv3, conv4], dim=1))
        upconv5 = self.up5(torch.cat([upconv4, conv3], dim=1))
        upconv6 = self.up6(torch.cat([upconv5, conv2], dim=1))
        upconv7 = self.up7(torch.cat([upconv6, conv1], dim=1))
        result = self.final_up(torch.cat([upconv7, initial], dim=1))
        return result
