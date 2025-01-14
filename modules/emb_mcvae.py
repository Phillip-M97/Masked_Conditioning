import torch
from torch import nn
from modules.globals import DEVICE


class ConditionMasking(nn.Module):
    '''
    Conditional masking is to be applied before embedding.
    With a probability p (altered by the sparsity scheduler) conditions are set to a predefined value.
    '''
    def __init__(self, arch: dict, spars_params: dict):
        super(ConditionMasking, self).__init__()
        mask_value = spars_params.get('mask_value', -1)
        self.mask_value = mask_value
        self.p = spars_params['sparsity']
        if self.p < 0.0 or self.p > 1.0:
            raise ValueError(f'sparsity has to be between 0 and 1, but got {self.p}')
        self.numerical_idx = [i for i, _ in enumerate(arch['cond_dims']) if arch['cond_dims'][i] in arch['numerical_cond']]
        self.numerical_idx_mask = torch.zeros(len(arch['cond_dims']), dtype=torch.bool)
        self.numerical_idx_mask[self.numerical_idx] = True

    def forward(self, y):
        assert len(y.shape) == 2, 'ConditionMasking expects tensors of shape [batch_size, num_conditions]'
        
        if not self.training or self.p == 0.0:
            y[:, ~self.numerical_idx_mask] += 1
            return y 
        
        mask = torch.rand_like(y) < self.p
        y[:, ~self.numerical_idx_mask] += 1
        
        y.masked_fill_(mask, 0)
        numerical_mask = mask[:, self.numerical_idx_mask]
        y[:, self.numerical_idx_mask].masked_fill_(numerical_mask, self.mask_value)
        return y


class ConditionEmbedding(nn.Module):
    '''
    This module embeds the inputs independently using a learned embedding layers.
    Categorical variables are embedded using the nn.Embedding module and numerical values using a linear layer.
    Each condition gets embedded into an independent vector of a fixed size. Then, embedded conditional vectors are concatenated to form a long conditioning vector.
    '''

    def __init__(self, arch: dict):
        super(ConditionEmbedding, self).__init__()
        self.numerical_conditions = arch['numerical_cond']
        self.cond_dims = arch['cond_dims']
        self.embedding_layers = nn.ModuleList()
        for i, _ in enumerate(arch['cond_dims']):
            if arch['cond_dims'][i] in self.numerical_conditions:
                self.embedding_layers.append(nn.Linear(1, arch['y_embed_dim']))
            else:
                self.embedding_layers.append(nn.Embedding(arch[arch['cond_dims'][i]]+1, arch['y_embed_dim'])) # +1 for case that value is masked
        self.numerical_indices = [i for i, dim in enumerate(self.cond_dims) if dim in self.numerical_conditions]
        self.categorical_indices = [i for i, dim in enumerate(self.cond_dims) if dim not in self.numerical_conditions]

    def forward(self, y):
        assert len(y.shape) == 2, 'ConditionEmbedding expects tensors of shape [batch_size, num_conditions]'
        # Note: embed all categorical variables using nn.Embedding, embed all numerical variables with nn.Linear (learned positional encoding)
        numerical_embeddings = [self.embedding_layers[i](y[:, i].view(-1, 1)) for i in self.numerical_indices]
        
        try:
            categorical_embeddings = [self.embedding_layers[i](y[:, i].long().view(-1, 1)).squeeze(1) for i in self.categorical_indices]
        except IndexError:
            for i in self.categorical_indices:
                cond = y[:, i].long().view(-1, 1)
                try:
                    self.embedding_layers[i](cond).squeeze(1)
                except IndexError:
                    print(f'Failed at condition {i} - {self.cond_dims[i]} with condition vector {cond} - Embedding layer has {self.embedding_layers[i].num_embeddings} entries')
                    raise IndexError
        return torch.cat(categorical_embeddings + numerical_embeddings, 1).to(DEVICE)


class mcVAE(nn.Module):
    '''
    The architecture is similar to the previous mcVAE but instead of using one-hot encoded vectors and positional encoded we use learned embeddings.
    Further, the masking is applied before the embedding such that masked values are embedded to a special vector.
    '''

    def __init__(self, arch: dict, spars_params: dict) -> None:
        super(mcVAE, self).__init__()
        self.msk = ConditionMasking(arch, spars_params)
        self.x_emb = self._emb_layers(arch)
        self.y_emb = ConditionEmbedding(arch)
        self.enc, self.dec = self._enc_dec_layers(arch)
        self.mean, self.logvar = self._mean_logvar_layers(arch)

    def _emb_layers(self, arch: dict) -> tuple:
        # unpack dimensions
        x_dim = arch['rp_dim']
        x_emb_dim = arch['rp_embed_dim']
        return nn.Linear(x_dim, x_emb_dim)

    def _enc_dec_layers(self, arch: dict) -> tuple:
        # unpack dimensions
        enc_input_dim = arch['rp_embed_dim']
        dec_input_dim = arch['latent_dim'] + len(arch['cond_dims']) * arch['y_embed_dim']
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
    
    def encode(self, x: torch.Tensor) -> tuple:
        x_enc = self.enc(x)
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

        # embedding of conditions
        y_msk = self.msk(y)
        y_emb = self.y_emb(y_msk)
        
        # embedding of x
        x_emb = self.x_emb(x)
        
        # encoding / downsampling path
        mean, logvar = self.encode(x_emb)
        z = self.reparameterization(mean, logvar)

        # decoding / upsampling path
        x_hat = self.decode(z, y_emb)

        return mean, logvar, y_emb, z, x_hat
    
class CarAdaptedConditionEmbedding(ConditionEmbedding):
    '''
    Expands on other CondtionEmbedding Module by using the same Embedding for all buzzword conditions.
    '''

    def __init__(self, arch: dict):
        super(CarAdaptedConditionEmbedding, self).__init__(arch)
        self.numerical_conditions = arch['numerical_cond']
        self.cond_dims = arch['cond_dims']
        self.embedding_layers = nn.ModuleList()
        
        if 'buzz_dim' in arch['cond_dims']:
            buzz_emb = nn.Embedding(arch['buzz_dim']+1, arch['y_embed_dim'])  # use the same embedding layer for all buzzword conditions
        for i, cond in enumerate(arch['cond_dims']):
            if arch['cond_dims'][i] in self.numerical_conditions:
                self.embedding_layers.append(nn.Linear(1, arch['y_embed_dim']))
            elif cond == 'buzz_dim':
                self.embedding_layers.append(buzz_emb)
            else:
                self.embedding_layers.append(nn.Embedding(arch[arch['cond_dims'][i]]+1, arch['y_embed_dim'])) # +1 for case that value is masked
        
        self.numerical_indices = [i for i, dim in enumerate(self.cond_dims) if dim in self.numerical_conditions]
        self.categorical_indices = [i for i, dim in enumerate(self.cond_dims) if dim not in self.numerical_conditions]
    
class CarAdaptedMCVae(mcVAE):
    '''
    Only difference to mcVAE is that CarAdaptedConditionEmbedding is used
    '''

    def __init__(self, arch: dict, spars_params: dict) -> None:
        super(CarAdaptedMCVae, self).__init__(arch, spars_params)
        self.y_emb = CarAdaptedConditionEmbedding(arch)