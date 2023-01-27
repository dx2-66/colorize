import numpy as np

class History:
    '''
    A struct for training/validation stats.
    '''
    def __init__(self, name='Unknown'):
        self.disc_loss = []
        self.gen_loss = []
        self.gen_loss_valid = []
        self.name = name
        
    def add(self, gen_loss, disc_loss=None, valid=False):
        '''
        Add record.
        '''
        if not valid:
            self.disc_loss.append(disc_loss)
            self.gen_loss.append(gen_loss)
        else:
            self.gen_loss_valid.append(gen_loss)
            
    def best_epoch(self):
        '''
        Best validation epoch for the generator.
        '''
        return np.argmin(np.array(self.gen_loss_valid))
        
    def display(self):
        '''
        Display stats.
        '''
        fig, ax = plt.subplots(1, 1, figsize=(16, 5))
        ax.set(xlim=(0, len(self.disc_loss)-1), ylim=(0, max(self.gen_loss_valid)), xlabel='Epoch')
        plt.plot(self.disc_loss);
        plt.plot(self.gen_loss);
        plt.plot(self.gen_loss_valid);
        plt.legend(['discriminator loss', 'generator loss (training)', 'generator loss(validation)']);
        print(f'{self.name}: best loss {min(self.gen_loss_valid):.4f} at epoch {self.best_epoch()+1}')
