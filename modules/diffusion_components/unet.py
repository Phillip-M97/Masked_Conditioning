import torch
from torch import nn

from modules.emb_mcvae import ConditionMasking, ConditionEmbedding
from .timeembedding import TimeEmbedding
from modules.conv_mcvae import SpatialSelfAttention, ResBlock, UpSample

""""
UpBlock, DownBlock and ResBlock implementations
Similar to implementation in conv_mcvae.py but differ in the way conditioning is integrated and by including time embeddings
Largely inspired by https://github.com/joh-fischer/PlantLDM which is a simplified version of Rombach et al.(2022)'s LDM without text conditioning.
"""

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

# time-conditional ResBlock extends standard VAE ResBlock by adding time conditioning to latent representation
class tcResBlock(ResBlock):
    
    def __init__(self, in_channels: int, out_channels: int, nonlinearity, time_emb_dim: int, dropout: float=0.0) -> None:
        super(tcResBlock, self).__init__(in_channels, out_channels, nonlinearity, dropout)
        self.time_emb = nn.Linear(time_emb_dim, out_channels)
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        #print('In Channels: ', self.in_channels)
        #print('X channels: ', x.size(1))
        h = self.norm1(x)
        h = self.nonlinearity(h)
        h = self.conv1(h)

        # add temporal embedding to latent representation
        # (only difference from VAE ResBlock)
        t_h = self.time_emb(t)
        h += t_h[:, :, None, None]

        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return self.shortcut(x) + h

class ConditionMapping(nn.Module):
    '''
    Condition Mapping maps a vector of size emb_dim*num_cond to an emb_dim vector
    Potentially applies normalization
    Reshapes the vector to a channels x width x height Tensor by repeating the vector for each pixel
    '''
    def __init__(self, y_emb_dim: int, num_conditions: int, norm: bool=True) -> None:
        super(ConditionMapping, self).__init__()
        self.map = nn.Linear(num_conditions*y_emb_dim, y_emb_dim)
        if norm:
            self.norm = nn.BatchNorm1d(y_emb_dim, affine=True)
        else:
            self.norm = nn.Identity()

    def forward(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        y = self.map(y)
        y = self.norm(y)
        y = y.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h.size(2), h.size(3))
        return y


class DiffusionDownBlock(nn.Module):
    
    def __init__(self, num_resblocks_per_level: int, in_channels: int, out_channels: int, time_emb_dim: int, y_emb_dim: int, num_conditions: int, nonlinearity, dropout: float=0.0, add_attention: bool=False, attention_heads: int=4, attention_dim: int=64) -> None:
        super(DiffusionDownBlock, self).__init__()
        assert num_resblocks_per_level > 0, 'Number of ResBlocks per level needs to be > 0'

        self.condition_mapping = ConditionMapping(y_emb_dim, num_conditions)

        self.res_blocks = nn.ModuleList()
        self.res_blocks.append(tcResBlock(in_channels+y_emb_dim, in_channels, nonlinearity, time_emb_dim, dropout))
        for _ in range(num_resblocks_per_level-1):
            self.res_blocks.append(tcResBlock(in_channels, in_channels, nonlinearity, time_emb_dim, dropout))
        
        if add_attention:
            self.res_blocks.append(
                SpatialSelfAttention(
                        channels=in_channels,
                        heads=attention_heads,
                        dim_head=attention_dim,
                        dropout=dropout
                    )
            )
        
        # downsample while increasing channels with strided convolution
        self.down_sample = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> tuple:
        # concatenate latent and conditioning information along channel dimension
        c_y = self.condition_mapping(y, x)
        h = torch.cat((x, c_y), dim=1)
        # apply each ResBlock and potentially attention
        for l in self.res_blocks:
            if isinstance(l, tcResBlock):
                h = l(h, t)
            else:
                h = l(h)
        # downsampling using MaxPooling
        downsampled = self.down_sample(h)
        return downsampled, h  # return downsampled and non-downsampled latent for Unet skips

class DiffusionUpBlock(nn.Module):
    
    def __init__(self, num_resblocks_per_level: int, in_channels: int, out_channels: int, time_emb_dim: int, y_emb_dim: int, num_conditions: int, nonlinearity, dropout: float=0.0, add_attention: bool=False, attention_heads: int=4, attention_dim: int=64):
        super(DiffusionUpBlock, self).__init__()
        assert num_resblocks_per_level > 0, 'Number of ResBlocks per level needs to be > 0'
        
        self.condition_mapping = ConditionMapping(y_emb_dim, num_conditions)

        self.res_blocks = nn.ModuleList()
        self.res_blocks.append(tcResBlock(in_channels+y_emb_dim, out_channels, nonlinearity, time_emb_dim, dropout))
        for _ in range(num_resblocks_per_level-1):
            self.res_blocks.append(tcResBlock(out_channels, out_channels, nonlinearity, time_emb_dim, dropout))
        
        if add_attention:
            self.res_blocks.append(
                SpatialSelfAttention(
                        channels=out_channels,
                        heads=attention_heads,
                        dim_head=attention_dim,
                        dropout=dropout
                    )
            )
        self.up_sample = UpSample(out_channels)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # concatenate latent and conditioning information along channel dimension
        c_y = self.condition_mapping(y, x)
        h = torch.cat((x, c_y), dim=1)
        for l in self.res_blocks:
            if isinstance(l, tcResBlock):
                h = l(h, t)
            else:
                h = l(h)
        return self.up_sample(h)

"""
Diffusion Model backend
Following Rombach et al. (2022) (--> LDM) we use a time-conditional Unet as the backbone for our diffusion model
Further, we include masked conditions according to our masked conditional generation scheme
"""
class mcUnet(nn.Module):

    def __init__(self, image_channels, model_channels, num_levels, num_resblocks_per_level, nonlinearity, time_emb_dim, pos_emb_dim, msk, y_emb, num_conditions, condition_dim, num_attention_heads, attention_dim, dropout: float=0.0, attention_levels: list=[]) -> None:
        super(mcUnet, self).__init__()

        self.msk = msk
        self.y_emb = y_emb
        self.time_embedding = TimeEmbedding(time_emb_dim, pos_emb_dim)
        self.x_emb = nn.Conv2d(image_channels, model_channels, kernel_size=3, stride=1, padding=1)

        # downward path
        self.enc = nn.ModuleList()
        curr_channels = model_channels
        next_channels = curr_channels*2
        for level in range(num_levels):
            self.enc.append(
                DiffusionDownBlock(
                    num_resblocks_per_level=num_resblocks_per_level,
                    in_channels=curr_channels,
                    out_channels=next_channels,
                    time_emb_dim=time_emb_dim,
                    y_emb_dim=condition_dim,
                    num_conditions=num_conditions,
                    nonlinearity=nonlinearity,
                    dropout=dropout,
                    add_attention=True if level in attention_levels else False,
                    attention_heads=num_attention_heads,
                    attention_dim=attention_dim
                )
            )
            curr_channels = next_channels
            next_channels = curr_channels*2
        
        # bottleneck
        self.mid_1 = tcResBlock(curr_channels, curr_channels, nonlinearity, time_emb_dim, dropout)
        self.mid_attn = SpatialSelfAttention(curr_channels, num_attention_heads, attention_dim, dropout)
        self.mid_2 = tcResBlock(curr_channels, curr_channels, nonlinearity, time_emb_dim, dropout)

        # upward path
        self.dec = nn.ModuleList()
        for level in reversed(list(range(num_levels))):
            next_channels = curr_channels//2
            self.dec.append(
                DiffusionUpBlock(
                    num_resblocks_per_level=num_resblocks_per_level,
                    in_channels=curr_channels*2,
                    out_channels=next_channels,
                    time_emb_dim=time_emb_dim,
                    y_emb_dim=condition_dim,
                    num_conditions=num_conditions,
                    nonlinearity=nonlinearity,
                    dropout=dropout,
                    add_attention=True if level in attention_levels else False,
                    attention_heads=num_attention_heads,
                    attention_dim=attention_dim
                )
            )
            curr_channels = next_channels

        # out conv
        self.out_conv = nn.Conv2d(in_channels=model_channels, out_channels=image_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, mask_y: bool=True):
        # print('Is training: ', self.training)
        # time embedding
        t = self.time_embedding(t)
        # embedding of masked conditions
        if mask_y:
            y_msk = self.msk(y.clone())
        else:
            y_msk = y.clone()
        y_e = self.y_emb(y_msk)

        # image channels to model channels
        h = self.x_emb(x)

        # downsample
        skips = []
        for i, block in enumerate(self.enc):
            h, predown = block(h, y_e, t)
            # print(f'Down Block {i}: current size: {predown.size()}, new size: {h.size()}, cond channels: 10')
            skips.append(predown)
        skips.append(h)
        
        # bottlenet
        h = self.mid_1(h, t)
        h = self.mid_attn(h)
        h = self.mid_2(h, t)

        # upsample
        for i, block in enumerate(self.dec):
            skip_conn = skips.pop()
            # print(f'Up Block {i}: skip size: {skip_conn.size()}, h size: {h.size()}, cond channels: 10')
            h = torch.cat((h, skip_conn), dim=1)
            h = block(h, y_e, t)
        
        # model_channels to image channels
        x_hat = self.out_conv(h)
        
        return x_hat