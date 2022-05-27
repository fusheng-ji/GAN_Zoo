import torch
from torch import nn
import torchvision
import numpy as np
import torch.nn.functional as F
# dataset
from torch.utils.data import DataLoader
from dataset import MNISTDataModule

# models
from models import Generator, Discriminator

# option
from opt import get_opts

# optimizer
from torch.optim import Adam

# pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
# log
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
import wandb
from collections import OrderedDict

class GAN(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        data_size = (1, 28, 28)
        self.generator = Generator(latent_dim = hparams.latent_dim, image_size = data_size)
        self.discriminator = Discriminator(image_size = data_size)
        
        self.validation_z = torch.randn(8, self.hparams.latent_dim)
        
        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)
        
        # wandb logger
        wandb.init(project="gan", entity="wenboji", name=hparams.exp_name)
        wandb.config = {
            "leaning_rate": hparams.lr,
            "epochs": hparams.num_epochs,
            "batch_size": hparams.batch_size,
            "latent_dim": hparams.latent_dim
        }
    def forward(self, z):
        return self.generator(z)
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch
        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)
        
        # train generator
        if optimizer_idx == 0:
            # gernerate images
            self.generated_imgs = self(z)
            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)
            
            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            wandb.log({"g_loss": g_loss,
                       
                       })
            return output
            
        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            wandb.log({"d_loss": d_loss,
                       "real_loss": real_loss,
                       "fake_loss": fake_loss,
                       })
            return output
    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.4, 0.8))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.4, 0.8))
        
        return [opt_g, opt_d], []
          
    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        wandb.log({"generated_images": wandb.Image(grid),
                   "step": self.current_epoch
                   })

if __name__ == '__main__':
    hparams = get_opts()
    dataset = MNISTDataModule(data_dir=hparams.data_dir, batch_size=hparams.batch_size)
    gan = GAN(hparams)
    
    trainer = Trainer(max_epochs=hparams.num_epochs,
                      enable_model_summary=True,
                      accelerator='auto',
                      devices=1,
                      num_sanity_val_steps=0, # test validation is right or false at ith step
                      log_every_n_steps=1,
                      check_val_every_n_epoch=20, # do validation behind every 20 epochs
                      benchmark=True)   
    trainer.fit(gan, dataset)
        