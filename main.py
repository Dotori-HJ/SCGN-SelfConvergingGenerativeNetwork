import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from config import get_config
from dataset import load_dataset
from models import Generator, weight_init, set_adam_state
from models.utils import make_interpolate
from ops import regularization_loss

if __name__ == '__main__':
    opt = get_config()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load Data
    train_data, train_loader = load_dataset(opt)

    # Load Checkpoint
    if os.path.exists(opt['checkpoint']) and os.path.exists(opt['z_data']):
        state = torch.load(opt['checkpoint'])
        create = False
    else:
        state = {}
        state['current_epoch'] = 0
        create = True

    # Tensorboard
    writer = SummaryWriter(opt['checkpoint_dir'])

    # Model define
    G = Generator(opt).to(device)
    G_optim = optim.Adam(G.parameters(), lr=opt['g_lr'])

    if create:
        G.apply(weight_init)
        print('Create Model')
    else:
        G.load_state_dict(state['G'])
        G_optim.load_state_dict(state['G_optim'])
        print('Model loaded')

    if torch.cuda.device_count() > 1:
        print('Use Multi GPU!')
        G = nn.DataParallel(G)

    mse_loss = nn.MSELoss()

    # Training
    fixed_Z = torch.randn(25, opt['nz']).to(device)
    for epoch in range(opt['epoch']):
        current_z_lr = opt['z_lr']

        k = 5e-2
        x0 = 200
        current_lambda = float(1 / (1 + np.exp(-k * (epoch - x0))))
        running_Z_loss = 0
        running_G_loss = 0
        running_kl_term = 0
        for idx, (X, Z_data, index) in enumerate(tqdm(train_loader)):
            bs = X.size()[0]
            X, Z_data = X.to(device), Z_data.to(device)

            Z, Z_optim = set_adam_state(Z_data, current_z_lr, epoch=epoch, betas=(0.9, 0.999), weight_decay=0)

            # Z update
            pred_X = G(Z)
            reg_loss = current_lambda * regularization_loss(Z.mean(dim=0), Z.std(dim=0))
            running_kl_term += reg_loss
            Z_loss = mse_loss(pred_X, X) + reg_loss

            Z_loss.backward()
            running_Z_loss += Z_loss.item()

            Z_optim.step()

            # G update
            G_optim.zero_grad()

            pred_X = G(Z.detach())
            G_loss = mse_loss(pred_X, X)
            G_loss.backward()
            running_G_loss += G_loss.item()

            G_optim.step()

            train_data.save(Z_data, index)

            if idx == 0:
                # save result images
                pred_X = pred_X.detach()
                save_image(X[:25], '{}/g_real_image.png'.format(opt['experiments_dir']), nrow=5, normalize=True)
                save_image(pred_X[:25], '{}/gen_image.png'.format(opt['experiments_dir']), nrow=5, normalize=True)

                with torch.no_grad():
                    random_Z = torch.randn(25, opt['nz']).to(device)
                    random_X = G(random_Z)
                    fixed_X = G(fixed_Z)
                save_image(random_X[:25], '{}/random_image.png'.format(opt['experiments_dir']), nrow=5, normalize=True)
                save_image(fixed_X[:25], '{}/fixed_image.png'.format(opt['experiments_dir']), nrow=5, normalize=True)

                images = make_interpolate(G, opt['nz'], device)
                save_image(images, '{}/interpolate_from_random.png'.format(opt['experiments_dir']), nrow=10,
                           normalize=True)
                images = make_interpolate(G, opt['nz'], device, Z[0].detach(), Z[1].detach())
                save_image(images, '{}/interpolate_from_z.png'.format(opt['experiments_dir']), nrow=10, normalize=True)

                if epoch % 10 == 0:
                    shutil.copy('{}/g_real_image.png'.format(opt['experiments_dir']),
                                '{}/g_real_image_{:d}.png'.format(opt['experiments_dir'], epoch))
                    shutil.copy('{}/random_image.png'.format(opt['experiments_dir']),
                                '{}/random_image_{:d}.png'.format(opt['experiments_dir'], epoch))
                    shutil.copy('{}/fixed_image.png'.format(opt['experiments_dir']),
                                '{}/fixed_image_{:d}.png'.format(opt['experiments_dir'], epoch))
                    shutil.copy('{}/gen_image.png'.format(opt['experiments_dir']),
                                '{}/gen_image_{:d}.png'.format(opt['experiments_dir'], epoch))
                    shutil.copy('{}/interpolate_from_random.png'.format(opt['experiments_dir']),
                                '{}/interpolate_from_random_{:d}.png'.format(opt['experiments_dir'], epoch))
                    shutil.copy('{}/interpolate_from_z.png'.format(opt['experiments_dir']),
                                '{}/interpolate_from_z_{:d}.png'.format(opt['experiments_dir'], epoch))
        losses = {
            'Z_loss': running_Z_loss / len(train_loader),
            'G_loss': running_G_loss / len(train_loader),
            'kl_term': running_kl_term / len(train_loader),
        }
        tqdm.write('epoch = {}/{}, Z_loss = {}, G_loss = {}'.format(epoch, opt['epoch'],
                                                                    running_Z_loss / len(train_loader),
                                                                    running_G_loss / len(train_loader)))

        writer.add_scalars('Losses', losses, global_step=epoch)

        if torch.cuda.device_count() > 1:
            net = None
            for module in G.children():
                net = module
            state['G'] = net.state_dict()
        else:
            state['G'] = G.state_dict()

        state['G_optim'] = G_optim.state_dict()
        train_data.commit()
        torch.save(state, opt['checkpoint'])
