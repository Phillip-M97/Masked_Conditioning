import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

class NLayerDiscriminator(nn.Module):
    """
    Implements a PatchGAN discriminator as withthe first stage of Stable Diffusion.
    Code is mainly taken from the Taming Transformers repository (https://github.com/CompVis/taming-transformers) which was inspired by the implementation for Pix2Pix (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
    """

    def __init__(self, img_channels: int=3, start_channels: int=64, n_layers: int=3):
        super(NLayerDiscriminator, self).__init__()

        layers = nn.ModuleList()
        # first layer differs from others
        layers.append(nn.Conv2d(in_channels=img_channels, out_channels=start_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, False))
        filter_multiplier = 1
        # gradual channel increase
        for n in range(1, n_layers):
            filter_multiplier_last = filter_multiplier
            filter_multiplier = min(2**n, 8)
            layers.append(
                nn.Conv2d(in_channels=start_channels*filter_multiplier_last, out_channels=start_channels*filter_multiplier, kernel_size=4, stride=2, padding=1, bias=False)
            )
            layers.append(
                nn.BatchNorm2d(start_channels*filter_multiplier, affine=True)
            )
            layers.append(nn.LeakyReLU(0.2, False))
        
        # last layer differs from others
        filter_multiplier_last = filter_multiplier
        filter_multiplier = min(2**n_layers, 8)
        layers.append(nn.Conv2d(start_channels*filter_multiplier_last, start_channels*filter_multiplier, kernel_size=4, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(start_channels * filter_multiplier, affine=True))
        layers.append(nn.LeakyReLU(0.2, False))
        # output layer is single channel predicition map
        layers.append(nn.Conv2d(start_channels*filter_multiplier, 1, kernel_size=4, stride=1, padding=1))

        self.main = layers
    
    def forward(self, x: Tensor) -> Tensor:
        h = x
        for i, l in enumerate(self.main):
            h = l(h)
        return h

'''
Implementation of adversarial loss, discriminator update and weights is from Taming Transformers.
'''

def adopt_weight(weight, global_step, threshold=0, value=0):
    if global_step < threshold:
        return value
    return weight

def adaptive_weight(vae_loss, d_loss, last_layer):
    vae_grad = torch.autograd.grad(vae_loss, last_layer, retain_graph=True)[0]
    d_grad = torch.autograd.grad(d_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(vae_grad) / (torch.norm(d_grad) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    return d_weight

# Discriminator loss is Hinge Loss
def disc_loss(logits_real: Tensor, logits_fake: Tensor):
    loss_real = F.relu(1. - logits_real).mean()
    loss_fake = F.relu(1. + logits_fake).mean()
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def discriminator_loss(x: Tensor, x_hat: Tensor, discriminator: NLayerDiscriminator, global_step: int, disc_factor: float=1, disc_start: int=0) -> Tensor:
    logits_real = discriminator(x.contiguous().detach())
    logits_fake = discriminator(x_hat.contiguous().detach())

    disc_factor = adopt_weight(disc_factor, global_step, threshold=disc_start)
    discriminator_loss = disc_factor * disc_loss(logits_real, logits_fake)

    return discriminator_loss

def adversarial_loss(x_hat: Tensor, discriminator: NLayerDiscriminator, vae_loss: Tensor, global_step: int, last_layer, disc_start: int=0, d_weight: float=1., disc_factor: float=1.) -> Tensor:
    logits_fake = discriminator(x_hat.contiguous())
    adv_loss = -logits_fake.mean()

    disc_factor = adopt_weight(disc_factor, global_step, threshold=disc_start)
    d_weight = d_weight * adaptive_weight(vae_loss, adv_loss, last_layer)
    
    """ if global_step % 200 == 0:
        print('disc_factor: ', disc_factor)
        print('d_weight = ', d_weight) """

    return d_weight * disc_factor * adv_loss

def step_disc(x: Tensor, x_hat: Tensor, discriminator: NLayerDiscriminator, optimizer: torch.optim.Optimizer, global_step: int, disc_start: int=0, d_weight: float=1.) -> Tensor:
    optimizer.zero_grad()
    d_loss = discriminator_loss(x, x_hat, discriminator, global_step, d_weight, disc_start)
    d_loss.backward()
    optimizer.step()
    return d_loss.detach()

