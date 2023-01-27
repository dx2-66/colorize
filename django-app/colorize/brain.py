import colorgram
from PIL import Image

import numpy as np

import torch
import torch.nn as nn

import albumentations as A
from albumentations.pytorch import ToTensorV2

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = torch.finfo(torch.float32).eps
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act='relu', use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode='reflect') if down else
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            PixelNorm(),
            nn.ReLU(inplace=True) if act=='relu' else nn.Mish(inplace=True),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

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
        
        self.embedding = nn.Linear(8 * 3, self.n_features)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(n_features*8, n_features*8, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
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
            rgb = torch.rand(8 * 3).repeat(x.shape[0],1)
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


def load_checkpoint(checkpoint_file, model):
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"])


def colorize(fss, generator, source, cgram=None):
    inference_transformations = A.Compose(
    [
        A.ToGray(),
        A.Resize(256,256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ])
    picture = lambda t: (t * 0.5 + 0.5).permute(1, 2, 0).cpu().detach().numpy()
    with Image.open(fss.path(source)) as image:
        width, height = image.size
        rgb = None if np.all(np.equal(cgram, None)) else torch.from_numpy(cgram).unsqueeze(0)
        image = np.array(image.convert("RGB"))
        input_tensor = inference_transformations(image=image)["image"]
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        with torch.no_grad():
            result = generator(input_batch, rgb)
            result = A.Compose([A.Resize(width=width, height=height, interpolation=3)])(image=picture(result[0]))["image"]
        return Image.fromarray((result * 255).astype('uint8'), 'RGB')

