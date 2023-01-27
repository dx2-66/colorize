from torch.utils.data import DataLoader


from mygan import model as gan
from mygan import config
from mygan.dataset import SplitDataset

print('Building a dataset, please wait warmly...')
train_set = SplitDataset(config.train_dir, 256)
test_set = SplitDataset(config.test_dir, 256)

train_loader = DataLoader(train_set, shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers)
test_loader = DataLoader(test_set, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True)

train_loader, test_loader = config.accelerator.prepare(train_loader, test_loader)

print('Building a model...')
model = gan.build_model()

print('Fitting...')
history = model.train(config.num_epochs, train_loader, test_loader)
