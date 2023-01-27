# Sketch colorization using Pix2Pix

Original paper: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)

Dataset used for training: https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair

Colorization is the one of the basic GAN tasks. This specific domain however introduces a tricky challenge: color palette tends to be very non-uniform, defined by the certain character design, or (in the cases of OC) limited only by the artist's imagination.

To address the issue without diverging from the original paper much, we'll be extracting the colormap from the image and pass it to the generator bottleneck. Given no colormap, it will be sampled from a uniform random distribution.

Though the original dataset comes with JSON colormaps, we decided to make the approach more flexible and extract colormaps as we go. Current implementation extracts 8 prevalent colors, it also allows to supply a reference color image. Though this is an imperfect solution (it breaks fully convolutional translation, and undithered extraction quality is questionable), this is about as far as we can go without major changes to the architecture.

Additionally, the model will be trained and tested on a stock task from the original paper (all the stock datasets: https://www.kaggle.com/datasets/vikramtiwari/pix2pix-dataset), 

**Required units:**

- PyTorch
- Albumentations
- 🤗 accelerate
- Pillow
- colorgram.py
- NumPy
- Matplotlib
- tqdm

**Contains:**

- An illustrated notebook with the complete code: *colorize-final.ipynb*.
- An importable unit implementation (*mygan* directory), complete with the simple examples (*train.py* and *run.py*).
- A django app, including the minimal generator implementation. Running live at https://pool.animeco.in/apps/colorize/

Download pretrained weights: [generator](https://pool.animeco.in/apps/static/MyGAN-gen-ann.pth), [discriminator](https://pool.animeco.in/apps/static/MyGAN-disc-ann.pth).

**Additional django app info:**

- app templates should be enabled;
- running torch applications via Apache WSGI requires `WSGIApplicationGroup %{GLOBAL}`. CPU is fine, colorgram extraction takes longer than a generator run.
