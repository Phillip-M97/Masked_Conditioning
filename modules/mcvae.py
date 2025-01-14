from modules.globals import *

import torch
import torch.nn as nn

class Mask(nn.Module):
    ''' Masks the condition by zeroing out inputs with probability (=sparsity).
        For unique conditions (manufacturer, type, class, drag coeff) it zeroes out a whole one hot encoded subvector.
        For non-unique conditions (buzzwords) it zeroes out single binary features of this one hot encoded subvector.
        In this way missing conditions are simulated, because that behaviour is desired in the inference.
    '''

    def __init__(self, arch: dict, spars_params: dict, mask_value: float=None) -> None:
        super().__init__()

        p = spars_params['sparsity']

        if p < 0.0 or p > 1.0:
            raise ValueError(f'sparsity has to be between 0 and 1, but got {p}')

        self.p = p
        self.dims = [
            arch[d] for d in arch['cond_dims']
        ]

        self.mask_value = mask_value

    def extra_repr(self) -> str:
        return f'p={self.p}'
    
    def forward(self, x: torch.Tensor, better: bool=False) -> torch.Tensor:
        assert x.shape[1] == sum(self.dims), 'input must be of shape (x, ' + str(sum(self.dims)) + ') but it is ' + str(x.shape)

        if not self.training or self.p == 0:
            return x
        
        # zero out single conditions
        # torch.manual_seed(42)
        if better:
            #this is better because it has a different mask per batch but unfortunately it is slow
            tight_mask = torch.rand((x.size(0), len(self.dims))) < self.p
            masks = [tight_mask[:, i].unsqueeze(1).expand(-1, dim) for i, dim in enumerate(self.dims)]
            full_mask = torch.cat(masks, dim=1).to(DEVICE)
        else:
            tight_mask = torch.rand(len(self.dims), device=DEVICE) > self.p

            # zero out full condition vectors depending on tight mask
            full_mask = torch.concat([
                torch.full((1, self.dims[i]), tight_mask[i].item()) for i in range(len(self.dims))
            ], dim=1).to(DEVICE)
        
        if self.mask_value is None:
            return x.where(full_mask, torch.zeros_like(x))
        else:
            return x.where(full_mask, torch.ones_like(x) * self.mask_value)

class PosEnc(nn.Module):
    r'''Applies a positional encoding to the input data like in the NeRF paper: https://dl.acm.org/doi/abs/10.1145/3503250'''
    
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        assert (out_features % 2 == 0 or out_features == in_features), 'dimension of encoding must be even or equal to in_features'
        self.in_features = in_features
        self.out_features = out_features
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, 'input must be 2d'
        assert x.shape[1] == self.in_features, 'shape 1 of given tensor is not equal to in_features'

        # if positional encoding is undesirable
        if self.out_features == self.in_features:
            return x
        
        dc_enc = torch.empty((x.shape[0], 0), dtype=torch.float, device=DEVICE)
        for i in range(self.out_features // 2):
            s = torch.sin(2**i * x * torch.pi)
            c = torch.cos(2**i * x * torch.pi)
            dc_enc = torch.concat([dc_enc, s, c], dim=1)
        return dc_enc

class mcVAE(nn.Module):
    ''' Architecture like cVAE but uses the Mask layer.
        Supports positional encoding layer for the drag coeff.
        This should make it possible to generate samples with incomplete conditions while inference.
    '''

    def __init__(self, arch: dict, spars_params: dict, do_positional_embed: bool=True, mask_value: float=None, better_masks: bool=True) -> None:
        super(mcVAE, self).__init__()

        self.msk = Mask(arch, spars_params, mask_value=mask_value)
        if do_positional_embed:
            self.p_enc = PosEnc(arch['dc_dim'], arch['dc_enc_dim'])
        else:
            self.p_enc = None
        self.x_emb, self.y_emb = self._emb_layers(arch)
        self.enc, self.dec = self._enc_dec_layers(arch)
        self.mean, self.logvar = self._mean_logvar_layers(arch)
        self.better_masks = better_masks

    def _emb_layers(self, arch: dict) -> tuple:

        # unpack dimensions
        x_dim = arch['rp_dim']
        y_dim = sum([arch[d] for d in arch['cond_dims']])
        x_emb_dim = arch['rp_embed_dim']
        y_emb_dim = arch['cond_embed_dim']

        return nn.Linear(x_dim, x_emb_dim), nn.Linear(y_dim, y_emb_dim)

    def _enc_dec_layers(self, arch: dict) -> tuple:

        # unpack dimensions
        enc_input_dim = arch['rp_embed_dim'] # + arch['cond_embed_dim']
        dec_input_dim = arch['latent_dim'] + arch['cond_embed_dim']
        hidden_dims = arch['hidden_dim']
        dec_output_dim = arch['rp_dim']
        activation = arch['activation']

        assert len(hidden_dims) > 0, 'there must be at least 1 hidden layer'

        # encoder
        enc = [nn.Linear(enc_input_dim, hidden_dims[0]), activation]
        for i in range(1, len(hidden_dims)):
            enc.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            enc.append(activation)

        # decoder
        dec = [nn.Linear(dec_input_dim, hidden_dims[-1]), activation]
        for i in range(len(hidden_dims)-1, 0, -1):
            dec.append(nn.Linear(hidden_dims[i], hidden_dims[i-1]))
            dec.append(activation)
        dec.append(nn.Linear(hidden_dims[0], dec_output_dim))

        return nn.Sequential(*enc), nn.Sequential(*dec)

    def _mean_logvar_layers(self, arch: dict) -> tuple:
        
        # unpack dimensions
        last_hidden_dim = arch['hidden_dim'][-1]
        latent_dim = arch['latent_dim']

        return nn.Linear(last_hidden_dim, latent_dim), nn.Linear(last_hidden_dim, latent_dim)

    def pos_enc(self, y: torch.Tensor) -> torch.Tensor:
        y_wo_dc = y[:,:-1]
        dc = y[:,-1].reshape(-1, 1)
        dc_enc = self.p_enc(dc)
        return torch.concat([y_wo_dc, dc_enc], dim=1)

    def mask(self, y: torch.Tensor) -> torch.Tensor:
        return self.msk(y, self.better_masks)

    def embed(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        return self.x_emb(x), self.y_emb(y)

    def encode(self, x: torch.Tensor) -> tuple:
        # xy = torch.concat([x, y], dim=1)
        # xy_enc = self.enc(xy)
        x_enc = self.enc(x)
        # mean, logvar = self.mean(xy_enc), self.logvar(xy_enc)
        mean, logvar = self.mean(x_enc), self.logvar(x_enc)
        return mean, logvar

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert (z.dim() == 2 & y.dim() == 2), 'input must be 2d'
        zy = torch.concat([z, y], dim=1)
        return self.dec(zy)

    def reparameterization(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(logvar, device=DEVICE)
        return mean + std * epsilon

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        assert (x.dim() == 2 & y.dim() == 2), 'input must be 2d'

        # positional encoding of drag coeff
        if self.p_enc:
            y_p_enc = self.pos_enc(y)
        else:
            y_p_enc = y

        # mask
        y_msk = self.mask(y_p_enc)
        
        # embedding
        x_emb, y_emb = self.embed(x, y_msk)
        
        # encode
        mean, logvar = self.encode(x_emb)
        z = self.reparameterization(mean, logvar)

        # decode
        x_hat = self.decode(z, y_emb)

        return mean, logvar, y_emb, z, x_hat