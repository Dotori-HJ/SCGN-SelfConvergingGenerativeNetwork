import torch.nn as nn

class _Block(nn.Module):
    def __init__(self, nf, upsample=False, downsample=False):
        super(_Block, self).__init__()
        self.upsample = upsample
        self.downsample = downsample

        modules = [
            nn.Conv2d(nf, nf, kernel_size=3, padding=1),
            nn.ELU(1.0, True),
            nn.Conv2d(nf, nf, kernel_size=3, padding=1),
            nn.ELU(1.0, True),
        ]
        if upsample:
            self.upsampler = nn.UpsamplingBilinear2d(scale_factor=2)
        if downsample:
            self.downsampler = nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1)
        self.main = nn.Sequential(*modules)

    def forward(self, input):
        res = self.main(input)

        if self.upsample:
            return self.upsampler(res + input)
        elif self.downsample:
            return self.downsampler(res + input)
        else:
            return res + input