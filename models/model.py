import math

import torch.nn as nn
from models.utils import View
from models.block import _Block


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        im_size = opt['im_size']
        embedding_size = opt['embedding_size']
        nc = opt['nc']
        nz = opt['nz']
        nf = opt['nf']
        nb = int(math.log(im_size, 2)) - int(math.log(embedding_size, 2))

        modules = [
            nn.Linear(nz, nf * embedding_size * embedding_size),
            View(-1, nf, embedding_size, embedding_size),
        ]
        for i in range(nb):
            modules.append(_Block(nf, upsample=True))
        modules.append(nn.Conv2d(nf, nc, kernel_size=3, padding=1))
        modules.append(nn.Tanh())
        self.main = nn.Sequential(*modules)

    def forward(self, input):
        return self.main(input)


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()

        im_size = opt['im_size']
        embedding_size = opt['embedding_size']
        nc = opt['nc']
        nz = opt['nz']
        nf = opt['nf']
        nb = int(math.log(im_size, 2)) - int(math.log(embedding_size, 2))

        modules = [
            nn.Conv2d(nc, nf, kernel_size=3, padding=1),
        ]
        for i in range(nb):
            modules.append(_Block(nf, downsample=True))
        removed = list(modules[-1].children())[:-1]
        removed.append(nn.AvgPool2d(kernel_size=2))
        modules[-1] = nn.Sequential(*removed)
        modules.append(View(-1, nf * embedding_size * embedding_size))
        modules.append(nn.Linear(nf * embedding_size * embedding_size, nz))

        self.main = nn.Sequential(*modules)

    def forward(self, input):
        return self.main(input)


if __name__ == '__main__':
    opt = {
        'im_size':64,
        'embedding_size':8,
        'nc':3,
        'nz':128,
        'nf':64,
    }

    g = Generator(opt)
    e = Encoder(opt, 'GAN')
    import torch
    vec = e(torch.randn(4, 3, 64, 64))
    print(vec.size())

    e = Encoder(opt, 'VAE')
    vec = e(torch.randn(4, 3, 64, 64))
    print(vec.size())

    vec = g(vec)
    print(vec.size())