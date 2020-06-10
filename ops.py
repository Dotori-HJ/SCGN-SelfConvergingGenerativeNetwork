import torch

def regularization_loss(z_mu, z_std):
   # KL(N(0,1)||z)
   return 0.5 * torch.sum(torch.log(z_std) - 1. + (1. + z_mu ** 2) / z_std)
