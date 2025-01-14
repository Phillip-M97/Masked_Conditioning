import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .emb_mcvae import ConditionEmbedding, ConditionMasking
from typing_extensions import Self

def closest_divisible(num_channels: int, num_groups: int) -> int:
    assert num_channels > 0 and num_groups > 0, 'Number of channels and groups must be larger than 0'

    if num_channels < num_groups:
        return num_channels

    if num_channels % num_groups == 0:
        return num_groups
    
    closest_multiple = num_channels//2
    for x in reversed(list(range(1, min(num_channels//2, num_groups)))):
        if num_channels % x == 0:
            return x
    
    return closest_multiple

class ResBlock(nn.Module):
    '''
    Implements BigGAN style ResBlocks with Group Normalization.
    Optionally, Dropout is applied.
    '''

    def __init__(self, in_channels: int, out_channels: int, nonlinearity, dropout: float=0.0) -> None:
        super().__init__()

        self.nonlinearity = nonlinearity
        self.in_channels = in_channels
        self.out_channels = out_channels

        num_groups = closest_divisible(in_channels, 32)
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        num_groups = closest_divisible(out_channels, 32)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.norm1(x)
        h = self.nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return self.shortcut(x) + h

class DiagonalGaussianDistribution:
    '''
    Implements a DiagonalGaussianDistribution which works for 2D data.
    Additionally provides utilities to sample from gaussian, calculate KL-divergence and determine the log-Likelihood of a sample
    '''
    
    def __init__(self, mean: torch.tensor, logvar: torch.tensor, deterministic: bool=False) -> None:
        self.mean = mean
        self.logvar = logvar
        self.deterministic = deterministic

        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(self.mean.device)

    # sample from distribution using reparametrization trick (therefore differentiable)
    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean).to(self.mean.device)
        return x

    # calculate KL-divergence beteen this distribution and either another Gaussian or the standard Gaussian
    def kld(self, other: Self=None, dims: list=[1, 2, 3]) -> torch.tensor:
        if self.deterministic:
            return torch.tensor([0.]).to(self.mean.device)
        if other is None:
            # if other is None return KLD to standard gaussian
            return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=dims)
        return 0.5 * torch.sum(torch.pow(self.mean - other.mean, 2) / other.var + self.var / other.var -1.0 - self.logvar + other.logvar, dim=dims)
    
    # return the negative log-likelihood of a sample under the current distribution
    def nll(self, sample, dims: list=[1, 2, 3]) -> torch.tensor:
        if self.deterministic:
            if sample == self.mean:
                return torch.tensor([1.0]).to(sample.device)
            else:
                return torch.tensor([0.]).to(sample.device)
        log_two_pi = np.log(2.0*np.pi)
        return 0.5 * torch.sum(log_two_pi + self.logvar, + torch.pow(sample - self.mean, 2)/self.var, dim=dims)
    
class DownBlock(nn.Module):
    '''
    A DownBlock represents one resolution level on the downward path.
    It consists of a predefined number of ResBlocks which are applied sequentially and a downsampling.
    Optionally, a SelfAttention module is applyed before downsampling the image.
    Downsampling is achieved here by applying a MaxPooling operation.
    '''

    def __init__(self, num_resblocks_per_level: int, in_channels: int, out_channels: int, nonlinearity, dropout: float, add_attention: bool, attention_kwargs: dict) -> None:
        super(DownBlock, self).__init__()
        assert num_resblocks_per_level > 0, 'Number of ResBlocks per level needs to be > 0'

        self.res_blocks = nn.ModuleList()
        self.res_blocks.append(ResBlock(in_channels, out_channels, nonlinearity, dropout))
        for _ in range(num_resblocks_per_level-1):
            self.res_blocks.append(ResBlock(out_channels, out_channels, nonlinearity, dropout))

        if add_attention:
            self.res_blocks.append(
                SpatialSelfAttention(
                        channels=out_channels,
                        heads=attention_kwargs.get('attention_heads', 4),
                        dim_head=attention_kwargs.get('attention_head_dim', 64),
                        dropout=dropout
                    )
            )
        # MaxPooling with kernelsize 2 and stride 2 halfs dimensionality of image in each direction
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for l in self.res_blocks:
            h = l(h)
        return self.down_sample(h)
    
class UpSample(nn.Module):
    '''
    This Module performs a learnable upsampling operation for the upward path of the net.
    Upsampling is achieved by first performing Nearest interpolation to increase the spatial dimensionality and then applying a convolution to improve the result.
    As the convolution weights are learnable the upsampling operation becomes a learnable operation.
    '''

    def __init__(self, channels: int):
        super(UpSample, self).__init__()
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        h = F.interpolate(x, (x.size(2)*2, x.size(3)*2), mode='nearest')
        return self.conv(h)
    
class UpBlock(nn.Module):
    '''
    An UpBlock represents one resolution level on the upward path.
    It consists of a predefined number of ResBlocks which are applied sequentially and a learnable upsampling operation.
    Optionally, a SelfAttention module is applyed before upsampling the image.
    Upsampling is achieved by using nearest interpolation and applying a convolution afterwards.
    '''
    def __init__(self, num_resblocks_per_level: int, in_channels: int, out_channels: int, nonlinearity, dropout: float, add_attention: bool=False, attention_kwargs: dict={}) -> None:
        super(UpBlock, self).__init__()
        assert num_resblocks_per_level > 0, 'Number of ResBlocks per level needs to be > 0'

        self.res_blocks = nn.ModuleList()
        self.res_blocks.append(ResBlock(in_channels, out_channels, nonlinearity, dropout))
        for _ in range(num_resblocks_per_level-1):
            self.res_blocks.append(ResBlock(out_channels, out_channels, nonlinearity, dropout))

        if add_attention:
            self.res_blocks.append(
                SpatialSelfAttention(
                    channels=out_channels,
                    heads=attention_kwargs.get('attention_heads', 4),
                    dim_head=attention_kwargs.get('attention_head_dim', 64),
                    dropout=dropout
                )
            )
        # learnable Upsampling doubles resolution along each spatial dimension
        self.up_sample = UpSample(out_channels)

    def forward(self, x):
        h = x
        for l in self.res_blocks:
            h = l(h)
        return self.up_sample(h)
    
class SpatialSelfAttention(nn.Module):
    """
    Implements Multi-Headed QKV Self Attention for spatial inputs
    Implementation inspired by Johannes Fischer (https://github.com/joh-fischer) who was inspired by the U-Net implementation for Diffusion by Ho et al. (https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py).
    """
    def __init__(self, channels: int, heads: int=4, dim_head: int=64, dropout: float=0.0) -> None:
        super(SpatialSelfAttention, self).__init__()
        self.channels = channels
        self.heads = heads
        self.dim_head = dim_head
        # for each head we need dim_head channels
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        num_groups = closest_divisible(channels, 32)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        # we need inner dim channels for Q, K and V
        self.qkv_mapping = nn.Conv1d(channels, self.inner_dim*3, kernel_size=1)
        self.proj_out = nn.Conv1d(self.inner_dim, self.channels, kernel_size=1)
        # zero out parameters to gradually learn influence
        for p in self.proj_out.parameters():
            p.detach().zero_()

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        try:
            res = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        except AttributeError:
            raise AttributeError("Please update to torch >2.0")
        return res
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, height, width = x.shape
        # flatten x
        x = x.reshape(b, c, -1)
        # get Q, K, V matrices
        qkv = self.qkv_mapping(self.norm(x))
        # we need to reshape to vectors (and back) as the PyTorch Attention implementation only works for vectors
        qkv = qkv.reshape(b, self.heads, qkv.shape[-1], -1)     # shape: (b, num_heads, h*w, 3*c)
        q, k, v = qkv.chunk(3, dim=-1)
        h = self.attention(q, k, v)
        h = h.reshape(b, self.inner_dim, -1)                        # shape: (b, nh*c, h*w)
        h = self.proj_out(h)                                    # shape: (b, c, h*w)
        # Skip Connection
        out = x + h
        return out.reshape(b, c, height, width)                 # restore original shape

    
class ConvMCVae(nn.Module):
    '''
    This Module adapts the embedding mcVAE used for parametric data to image data using convolutions in residual blocks.
    Essentially we downscale an image in the encoding path while increasing the channels.
    Then, channels are reduced in the bottleneck to create the low-dim embedding.
    We sample from a Gaussian distribution using the embedding as means and variances to implement variational inference.
    The (masked) conditional information is concatenated to the embedding.
    Then the number of channels in the embedding is increased again.
    In the decoding path the conditioned latent embedding is upsampled by reducing the number of channels to create the reconstructed original image.
    '''

    def __init__(self, arch: dict, spars_params: dict) -> None:
        super(ConvMCVae, self).__init__()
        self.msk = ConditionMasking(arch, spars_params)

        # embedding layers
        self.x_emb = nn.Conv2d(arch['image_channels'], arch['im_embed_dim'], kernel_size=3, stride=1, padding=1)
        self.y_emb = ConditionEmbedding(arch)

        # encoding path
        curr_channels = arch['im_embed_dim']
        next_channels = arch['start_channels']
        self.enc = nn.ModuleList()
        for level in range(arch['num_levels']):
            self.enc.append(
                DownBlock(
                        num_resblocks_per_level=arch['num_blocks_per_level'],
                        in_channels=curr_channels,
                        out_channels=next_channels,
                        nonlinearity=arch['nonlinearity'],
                        dropout=arch['dropout'],
                        add_attention=True if level in arch.get('attention_levels', []) else False,
                        attention_kwargs=arch
                    )
                )
            curr_channels = next_channels
            next_channels = curr_channels*2

        # bottleneck
        condition_channels = arch.get('condition_channels', arch['y_embed_dim'])
        self.condition_mapping = nn.Linear(len(arch['cond_dims'])*arch['y_embed_dim'], condition_channels)
        self.condition_norm = nn.BatchNorm1d(condition_channels)

        self.bottleneck_in = ResBlock(curr_channels, arch['latent_dim']*2, arch['nonlinearity'], arch['dropout'])
        self.bottleneck_out = ResBlock(arch['latent_dim'] + condition_channels, curr_channels, arch['nonlinearity'], arch['dropout'])

        # decoding path
        self.dec = nn.ModuleList()
        for level in reversed(list(range(arch['num_levels']))):
            next_channels = curr_channels//2
            self.dec.append(
                UpBlock(
                    num_resblocks_per_level=arch['num_blocks_per_level'],
                    in_channels=curr_channels,
                    out_channels=next_channels,
                    nonlinearity=arch['nonlinearity'],
                    dropout=arch['dropout'],
                    add_attention=True if level in arch.get('attention_levels', []) else False,
                    attention_kwargs=arch
                )
            )
            curr_channels = next_channels

        # out convolution to get image with correct number of channels (e.g. 1 for greyscale, 3 for RGB)
        num_groups = closest_divisible(curr_channels, 32)
        self.norm_out = nn.GroupNorm(num_channels=curr_channels, num_groups=num_groups, eps=1e-6, affine=True)
        self.out = nn.Conv2d(curr_channels, arch['image_channels'], kernel_size=3, stride=1, padding=1)

    def embed_y(self, z, y):
        # embed y vector using learned embedding layers
        y_emb = self.y_emb(y)
        # map to predefined length for efficiency and normalize to control KLD
        y_emb = self.condition_mapping(y_emb)
        y_emb = self.condition_norm(y_emb)
        # reshape vector to (emb_dim, width, height) to match latent
        y_emb = y_emb.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, z.size(2), z.size(3))
        return y_emb

    def encode(self, x):
        h = self.x_emb(x)
        for l in self.enc:
            h = l(h)
        z = self.bottleneck_in(h)
        return z

    def reparametrization(self, x):
        # first half of channels are means, second half of channels are (log-)variances
        mean, logvar = torch.chunk(x, 2, dim=1)
        dist = DiagonalGaussianDistribution(mean, logvar)
        # sample using reparametrization trick
        return dist.sample(), mean, logvar
    
    def decode(self, x, y):
        xy = torch.cat([x, y], dim=1)
        h = self.bottleneck_out(xy)
        for l in self.dec:
            h = l(h)
        h = self.norm_out(h)
        x_hat = self.out(h)
        return x_hat

    def forward(self, x, y):
        # mask conditions randomly
        y_msk = self.msk(y)

        # encoding path
        h = self.encode(x)
        # sampling
        z, mean, logvar = self.reparametrization(h)

        # embed conditions and reshape to appropriate shape
        y_emb = self.embed_y(z, y_msk)

        # conditional reconstruction
        x_hat = self.decode(z, y_emb)
        return mean, logvar, y_emb, z, x_hat
    