import argparse
import os


def get_config():
    parser = argparse.ArgumentParser(description='SCGN:Self Converging Generative Network.')
    parser.add_argument('--dataset', type=str,
                        choices=['mnist', 'cifar10', 'cifar100', 'celeba', 'lsun'],
                        help='name of dataset cnndnnrnngto use [mnist, cifar10]')
    parser.add_argument('--train', dest='train', action='store_true', help='train network')
    parser.add_argument('--epoch', type=int, default=1000, help='input total epoch')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--g_lr', type=float, default=1e-4)
    parser.add_argument('--z_lr', type=float, default=1e-1)
    parser.add_argument('--nz', type=int, default=128)
    parser.add_argument('--nf', type=int, default=64)
    parser.add_argument('--lambda', type=float, default=1e-8)
    parser.add_argument('--embedding_size', type=int, default=8)

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--experiments_dir', type=str, default='began_experiments')
    parser.add_argument('--checkpoint_dir', type=str, default='began_checkpoint')
    parser.add_argument('--z_data_dir', type=str, default='began_z_data')
    opt = vars(parser.parse_args())
    init(opt)
    return opt


def init(opt):
    path = os.path.join(opt['experiments_dir'], opt['dataset'])
    if not os.path.exists(path):
        os.makedirs(path)
    opt['experiments_dir'] = path

    path = os.path.join(opt['z_data_dir'], opt['dataset'])
    if not os.path.exists(path):
        os.makedirs(path)
    opt['z_data_dir'] = path

    path = os.path.join(opt['z_data_dir'], 'z_data.db')
    opt['z_data'] = path

    path = os.path.join(opt['checkpoint_dir'], opt['dataset'])
    if not os.path.exists(path):
        os.makedirs(path)
    opt['checkpoint_dir'] = path

    path = os.path.join(opt['checkpoint_dir'], 'checkpoint.pth')
    opt['checkpoint'] = path
