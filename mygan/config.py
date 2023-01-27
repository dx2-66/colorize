from accelerate import Accelerator

batch_size = 16 # recommended: biggest multiple of 8 memory can fit for batch processing
num_workers = 12 # recommended: either 4 * number of GPUs or number of CPU cores
num_epochs = 10
colormap_size = 8
train_dir = '/home/daiyousei/storage/datasets/cityscapes/cityscapes/train/'
test_dir = '/home/daiyousei/storage/datasets/cityscapes/cityscapes/val/'
image_size = 512

accelerator = Accelerator()
device = accelerator.device
