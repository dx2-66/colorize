import torch
import torch.nn as nn

import albumentations as A
from PIL import Image
import numpy as np

from mygan import config

picture = lambda t: (t * 0.5 + 0.5).permute(1, 2, 0).cpu().detach().numpy()

pre_extraction = lambda filename: Image.fromarray(A.Compose(
    [
        A.Resize(width=256, height=256, interpolation=2)
    ])
    (image=np.array(Image.open(filename).convert("RGB")))["image"])

pre_extraction_left_half = lambda filename: Image.fromarray(A.Compose(
    [
        A.SmallestMaxSize (max_size=256, interpolation=2),
        A.Crop(x_max=256, y_max=256)
    ])
    (image=np.array(Image.open(filename).convert("RGB")))["image"])

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = torch.finfo(torch.float32).eps
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

# Save/load routines: 
def save_checkpoint(model, optimizer, filename):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer):
    checkpoint = torch.load(checkpoint_file, map_location=config.device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
