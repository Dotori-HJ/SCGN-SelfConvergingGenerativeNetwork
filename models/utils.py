import torch
import torch.nn as nn
import torch.optim as optim


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)


def weight_init(m):
    mean, std = 0.0, 0.02
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=mean, std=std)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 1.0, std)
        torch.nn.init.constant_(m.bias, 0.0)


def set_adam_state(Z_data, lr, epoch, betas=(0.9, 0.999), weight_decay=0):
    Z = Z_data[:, 0].requires_grad_()

    optimizer = optim.Adam((Z,), lr=lr, betas=betas, weight_decay=weight_decay)
    for group in optimizer.param_groups:
        p = group['params'][0]
        s = optimizer.state[p]
        s['step'] = epoch
        s['exp_avg'] = Z_data[:, 1]
        s['exp_avg_sq'] = Z_data[:, 2]

    return Z, optimizer


def get_adam_state(optimizer):
    for group in optimizer.param_groups:
        p = group['params'][0]

        s = optimizer.state[p]

        return s


def set_sgd_state(Z_data, lr, momentum=0.9, weight_decay=0):
    Z = Z_data[:, 0].requires_grad_()

    optimizer = optim.SGD((Z,), lr=lr, momentum=momentum, weight_decay=weight_decay)
    for group in optimizer.param_groups:
        p = group['params'][0]
        s = optimizer.state[p]
        s['momentum_buffer'] = Z_data[:, 1]

    return Z, optimizer


def make_interpolate(model, z_dim, device, start=None, end=None):
    z = []
    if start is None and end is None:
        z0 = torch.randn(z_dim).to(device)
        z9 = torch.randn(z_dim).to(device)
    else:
        z0 = start
        z9 = end

    z.append(z0)
    for i in range(1, 9):
        alpha = i * 0.1111
        point = (1 - alpha) * z0 + alpha * z9
        z.append(point)
    z.append(z9)
    z = torch.cat([z_i.view(1, z_dim) for z_i in z], dim=0)

    images = model(z)
    return images


def slerp(val, low, high):
    omega = torch.acos(torch.dot(low / torch.norm(low), high / torch.norm(high)))
    so = torch.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
    return torch.sin((1.0 - val) * omega) / so * low + torch.sin(val * omega) / so * high