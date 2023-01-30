import torch
from torch.utils.data import Dataset

import numpy as np
import colorgram

from PIL import Image
from pathlib import Path
from multiprocessing import Pool


from mygan import config, augmentation
from mygan.util import pre_extraction_left_half, pre_extraction_right_half

class SplitDataset(Dataset):
    '''
    Sketch-colorized image pairs vertically stacked in a single file.
    '''
    @staticmethod
    def process_dir(path, target_on_right=False):
        '''
        Extracts the observation id and the colorgram.
        '''
        if target_on_right:
            return (path.name, colorgram.extract(pre_extraction_right_half(path), config.colormap_size))
        else:    
            return (path.name, colorgram.extract(pre_extraction_left_half(path), config.colormap_size))
    
    def __init__(self, root, size=config.image_size, target_on_right=False):
        '''
        Accepts the dataset directory. Prepares a file list and colorgram list.
        Cologram extraction is a resource-heavy operation and utilizes multiprocessing.
        Estimated initialization time: 10 min (Ryzen 3600X, 12 threads, M2 SSD).
        Default individual image size is assumed to be 512x512.
        '''
        self.root = root
        self.size = size
        self.target_on_right = target_on_right
        self.files = []
        # build a list of files:
        pool = Pool(config.num_workers)
        path1 = Path(root)
        top = [dir for dir in filter(Path.is_file, path1.iterdir())]
        results = pool.map(SplitDataset.process_dir, top, self.target_on_right)
        if results:
            self.files.extend(results)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filename, cgram = self.files[idx]
        path = Path(self.root, filename)
        image = np.array(Image.open(path))
        
        # Split:
        if self.target_on_right:
            source = image[:, :self.size, :]
            target = image[:, self.size:, :]
        else:
            target = image[:, :self.size, :]
            source = image[:, self.size:, :]
        
        # Augment:
        augmentations = augmentation.both_transform(image=source, image0=target)
        source = augmentations["image"]
        target = augmentations["image0"]

        source = augmentation.transform_only_input(image=source)["image"]
        target = augmentation.transform_only_mask(image=target)["image"]
        
        # Pass the cologram further as a flattened array:
        rgb = np.array([c.rgb.r for c in cgram] + [c.rgb.g for c in cgram] + [c.rgb.b for c in cgram]).astype('float32') / 255.
        rgb = np.pad(rgb, (0, config.colormap_size * 3 - len(rgb)), constant_values=1)
        return source, target, torch.from_numpy(rgb)
