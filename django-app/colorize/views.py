from django.shortcuts import render

from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings


import colorgram
import albumentations as A
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import os

from . import brain


generator = brain.Generator()
brain.load_checkpoint(os.path.join(settings.STATIC_ROOT,'MyGAN-gen-ann.pth'), generator)

def extract_colors(colormap_file, fss):
                '''
                Input: an image file and FileSystemStorage object.
                Returns: pallette image URL and color array for the generator.
                '''
                colormap = colorgram.extract(Image.fromarray(A.Compose([A.Resize(width=256, height=256, interpolation=2)])(image=np.array(Image.open(fss.path(colormap_file)).convert("RGB")))["image"]), 8)
                #colormap = colorgram.extract(Image.open(fss.path(colormap_file)), 8)
                rgb = np.array([c.rgb.r for c in colormap] + [c.rgb.g for c in colormap] + [c.rgb.b for c in colormap]).astype('float32') / 255.
                rgb = np.pad(rgb, (0, 24-len(rgb)), constant_values=1)
                colors = np.split(rgb, 8)
                fig = plt.figure(figsize=(10, 1))
                ax = fig.add_subplot(111)

                for i, color in enumerate(colors):
                    ax.barh([''], 1, align='center', height=0.2, left=i, color=color)

                plt.axis('off')
                palette_name = fss.get_available_name('graph.png')
                saveto = fss.path(palette_name)
                plt.savefig(saveto, bbox_inches='tight', transparent="True", pad_inches=0)
                return fss.url(palette_name), rgb

def upload(request):
    payload = {}
    if request.method == 'POST' and request.FILES:
        # Get source file:
        if request.FILES.get('source'):
            source = request.FILES['source']
            fss = FileSystemStorage()
            source_file = fss.save(source.name, source)
            source_url = fss.url(source_file)
            payload['source_url'] = source_url
            # Extract colors from reference, if any:
            if request.FILES.get('colormap'):
                colormap = request.FILES['colormap']
                colormap_file = fss.save(colormap.name, colormap)
                palette_name, rgb = extract_colors(colormap_file, fss)
                payload['colormap_url'] = palette_name
                # Pass the source to the model:
                target = brain.colorize(fss, generator, source_file, rgb)

            else:
                target = brain.colorize(fss, generator, source_file)

            # Render the result and the final colormap:
            target_name = fss.get_available_name('result.png')
            saveto = fss.path(target_name)
            target.save(saveto)
            payload['target_url'] = fss.url(target_name)

            palette_name2, _ = extract_colors(saveto, fss)
            payload['out_colormap_url'] = palette_name2

    return render(request, 'main.html', payload)
