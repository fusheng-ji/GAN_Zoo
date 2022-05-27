import torch
from torch import nn

import numpy as np

class Generator(nn.Module):
    def __init__(self, latent_dim, image_size):
         super().__init__()
         self.image_size = image_size
         self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.GELU(),
            nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.GELU(),
            nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.GELU(),
            nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.GELU(),
            nn.Linear(1024, np.prod(image_size, dtype=np.int32)),
            nn.Sigmoid(),
        ) 
    def forward(self, z):
        # shape of z: [batchsize, latent_dim]
        output = self.model(z)
        image = output.view(output.size(0), *self.image_size)

        return image
    
class Discriminator(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size
        self.model = nn.Sequential(
                nn.Linear(np.prod(self.image_size, dtype=np.int32), 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )    
    def forward(self, image):
        # shape of image:[batchsize, 1, 28, 28]
        validity = self.model(image.reshape(image.shape[0], -1))
        
        return validity
    