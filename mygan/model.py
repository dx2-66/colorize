import torch
import torch.nn as nn
from torch.optim import Adam, AdamW

from tqdm import tqdm

from mygan.discriminator import Discriminator
from mygan.generator import Generator
from mygan.util import save_checkpoint
from mygan.history import History
from mygan import config


class MyGAN:
    def __init__(self, discriminator, generator, lr=5e-4, ratio=5.):
        disc_optim = AdamW(discriminator.parameters(), lr=lr/ratio, betas=(0.5, 0.999))
        gen_optim = AdamW(generator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        self.discriminator, self.generator, self.disc_optim, self.gen_optim = config.accelerator.prepare(
                                                                                                    discriminator,
                                                                                                    generator,
                                                                                                    disc_optim,
                                                                                                    gen_optim,
                                                                                                 )
        self.criterion = nn.BCEWithLogitsLoss()
        self.similarity = nn.L1Loss()
        
    def train(self, epochs, data_tr, data_val):
        # Forward declaration:
        return train_gan(self, epochs, data_tr, data_val)


# Optional: DCGAN-style weight initialization, seems to improve the stability a bit.
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def train_gan(model, epochs, data_tr, data_val):
    title = f'{type(model).__name__}'
    history = History(title)
    disc_sched = torch.optim.lr_scheduler.CosineAnnealingLR(model.disc_optim, T_max=epochs, eta_min=1e-7)
    gen_sched = torch.optim.lr_scheduler.CosineAnnealingLR(model.gen_optim, T_max=epochs, eta_min=1e-7)

    for epoch in range(epochs):
        print('* Epoch %d/%d' % (epoch+1, epochs))

        loss_disc = 0
        loss_gen = 0

        for source, target, cgram in tqdm(data_tr):
            model.discriminator.train()
            model.generator.train()
            
            # Colorize:
            fake = model.generator(source, cgram)
            # Check the discriminator against real and generated images:
            estimation_real = model.discriminator(source, target)
            estimation_fake = model.discriminator(source, fake.detach())
            real_loss = model.criterion(estimation_real, torch.ones_like(estimation_real))
            fake_loss = model.criterion(estimation_fake, torch.zeros_like(estimation_fake))
            # Average loss:    
            disc_loss = (real_loss + fake_loss) / 2.
            model.disc_optim.zero_grad()
            config.accelerator.backward(disc_loss)
            model.disc_optim.step()
            loss_disc += disc_loss
            
            # Check the generator result against the discriminator:
            estimation_fake = model.discriminator(source, fake)
            fake_loss = model.criterion(estimation_fake, torch.ones_like(estimation_fake))
            # Check the generator result against the ground truth:
            target_loss = model.similarity(fake, target) * 100.
            # Weighted sum:
            gen_loss = fake_loss + target_loss
        
            model.gen_optim.zero_grad()
            config.accelerator.backward(gen_loss)
            model.gen_optim.step()
            loss_gen += gen_loss
        
        disc_sched.step()
        gen_sched.step()
        
        loss_disc /= len(data_tr)
        loss_gen /= len(data_tr)    
        history.add(loss_gen.detach().cpu(), loss_disc.detach().cpu())
        print(f'Epoch {epoch+1}:\nTraining: disc loss {loss_disc:.2f}, gen loss {loss_gen:.2f}')            

        model.discriminator.eval()
        model.generator.eval()
        
        loss_gen = 0
        
        # Validation:
        with torch.no_grad():
            for source, target, cgram in tqdm(data_val):
                fake = model.generator(source, cgram)
                estimation_fake = model.discriminator(source, fake)
                fake_loss = model.criterion(estimation_fake, torch.ones_like(estimation_fake))
                target_loss = model.similarity(fake, target) * 100.
        
                gen_loss = fake_loss + target_loss
                loss_gen += gen_loss
            
            loss_gen /= len(data_val)
            history.add(loss_gen.detach().cpu(), valid=True)

        print(f'Validation: disc loss {loss_disc:.2f}, gen loss {loss_gen:.2f}');
        
        if epoch == history.best_epoch():
            save_checkpoint(model.discriminator, model.disc_optim, f'{title}-disc.pth')
            save_checkpoint(model.generator, model.gen_optim, f'{title}-gen.pth')
    torch.cuda.empty_cache()
    model.discriminator.eval()
    model.generator.eval()
    return history
    

def build_model():
    '''
    Assembles the network.
    '''
    discriminator = Discriminator()
    generator = Generator()
    
    initialize_weights(discriminator)
    initialize_weights(generator)
    
    model = MyGAN(discriminator, generator)
    
    return model
