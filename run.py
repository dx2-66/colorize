import torch
import numpy as np
from PIL import Image
import colorgram
import argparse


from mygan import model as gan
from mygan.augmentation import inference_transform, upscale
from mygan.util import load_checkpoint, picture
from mygan import config


def colorize(source, cgram=None):
    '''
    Accepts the source and optional colorgram filenames,
    returns the resulting PIL image.
    '''
    rgb = None
    with Image.open(source) as image:
        width, height = image.size
        if cgram:
            cgram = colorgram.extract(Image.open(cgram), config.colormap_size)
            rgb = np.array([c.rgb.r for c in cgram] + [c.rgb.g for c in cgram] + [c.rgb.b for c in cgram]).astype('float32') / 255.
        image = np.array(image.convert("RGB"))
        input_tensor = inference_transform(image=image)["image"]
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        with torch.no_grad():
            # Numpy arrays have no single truth value:
            if not np.all(np.equal(rgb, None)):
                result = model.generator(input_batch.to(config.device), torch.from_numpy(rgb).unsqueeze(0).to(config.device))
            else:
                result = model.generator(input_batch.to(config.device))
            result = upscale(width, height)(image=picture(result[0]))["image"]
        return Image.fromarray((result * 255).astype('uint8'), 'RGB')
        
parser = argparse.ArgumentParser()
parser.add_argument('input', help='source image file', type=str)
parser.add_argument('colorgram', help='optional colorgram file', type=str, nargs='?')
args = parser.parse_args()

model = gan.build_model()

load_checkpoint('MyGAN-gen.pth', model.generator, model.gen_optim)

output = colorize(args.input, args.colorgram)

output.save('output.png')
