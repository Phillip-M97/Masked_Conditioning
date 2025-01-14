import torch
from torch import nn, Tensor
from .globals import DEVICE
from .emb_mcvae import ConditionEmbedding, ConditionMasking
from .conv_mcvae import DownBlock, UpBlock


class Encoder(nn.Module):
    '''
    The encoders goal is to downsample the image sequentially and map it to a latent code.
    The latent code is then decoded to an image by the Decoder.
    Further, the encoders latent codes are to be matched by the MaskedLatentMapper.
    '''

    def __init__(self, latent_dim: int, start_channels: int, n_levels: int, image_channels: int=3, image_width: int=256, image_height: int=256, num_resblocks_per_level: int=2, attention_levels: list=[2], attention_kwargs: dict={'attention_heads': 4, 'attention_head_dim': 64}):
        super(Encoder, self).__init__()

        self.start_conv = nn.Conv2d(image_channels, start_channels, kernel_size=3, stride=1, padding=1)

        curr_channels = start_channels
        layers = []
        for level in range(n_levels):
            next_channels = curr_channels*2
            layers.append(
                DownBlock(
                    num_resblocks_per_level=num_resblocks_per_level,
                    in_channels=curr_channels,
                    out_channels=next_channels,
                    nonlinearity=nn.LeakyReLU(0.1),
                    dropout=0.0,
                    add_attention=True if level in attention_levels else False,
                    attention_kwargs=attention_kwargs
                )
            )
            curr_channels = next_channels
        self.main = nn.Sequential(*layers)

        final_dim = curr_channels * (image_width // (2**n_levels)) * (image_height // (2**n_levels))
        self.final_norm = nn.BatchNorm2d(curr_channels, affine=True)
        self.latent_mapper_1 = nn.Linear(final_dim, final_dim//4)
        self.latent_norm = nn.BatchNorm1d(final_dim//4, affine=True)
        self.latent_mapper = nn.Linear(final_dim//4, latent_dim)
        self.latent_act = nn.Tanh()
    
    def forward(self, x: Tensor):
        h = self.start_conv(x)
        h = self.main(h)
        h = self.final_norm(h)
        h = h.reshape(h.size(0), -1)
        # fc part
        z = self.latent_mapper_1(h)
        z = self.latent_norm(z)
        z = self.latent_mapper(z)
        z = self.latent_act(z)
        return z


class Generator(nn.Module):
    '''
    The generator module takes a latent code and noise and maps it to a reconstructed image.
    At training time the latent code is produced by the Encoder network.
    However, at inference timt the latent code is produced by the MaskedLatentMapper.
    '''

    def __init__(self, latent_dim: int, noise_dim: int, start_channels: int, n_levels: int, image_width: int, image_height: int, image_channels: int=3, num_resblocks_per_level: int=2, attention_levels: list=[2], attention_kwargs: dict={'attention_heads': 4, 'attention_head_dim': 64}):
        super(Generator, self).__init__()

        self.start_pixels_w = image_width // (2**n_levels)
        self.start_pixels_h = image_height // (2**n_levels)

        curr_channels = start_channels*(2**n_levels)
        self.start_channels = curr_channels
        self.latent_mapper = nn.Linear(latent_dim + noise_dim, curr_channels*self.start_pixels_w*self.start_pixels_h)
        self.latent_norm = nn.BatchNorm1d(curr_channels*self.start_pixels_w*self.start_pixels_h, affine=True)

        layers = []
        for level in reversed(list(range(n_levels))):  # remove last [:1]
            next_channels = curr_channels//2
            layers.append(
                    UpBlock(
                        num_resblocks_per_level=num_resblocks_per_level,
                        in_channels=curr_channels,
                        out_channels=next_channels,
                        nonlinearity=nn.LeakyReLU(0.1),
                        dropout=0.0,
                        add_attention=True if level in attention_levels else False,
                        attention_kwargs=attention_kwargs
                    )
                )
            curr_channels = next_channels

        # last level is only transpose conv
        # layers.append(nn.ConvTranspose2d(curr_channels, image_channels, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.Conv2d(curr_channels, image_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)
    
    def forward(self, z: Tensor, n: Tensor) -> Tensor:
        zn = torch.cat([z, n], dim=1)
        h = self.latent_mapper(zn)
        h = self.latent_norm(h)
        h = h.reshape(zn.size(0), self.start_channels, self.start_pixels_h, self.start_pixels_w)       
        x_hat = self.main(h)
        return x_hat


class MaskedLatentMapper(nn.Module):
    '''
    The MaskedLatentMapper maps a random sample z and a set of conditions y to a latent code
    The sequential part of the model is taken from the original publication.
    In addition we use embedding layers to embed the conditions.
    The concatenated conditioning vectors are projected to a lower dimensionality and a BatchNormalization is applied as it provided superior performance in our previous tests.
    '''

    def __init__(self, msk: ConditionMasking, emb: ConditionEmbedding, cond_dim: int, num_conds: int, latent_dim: int, model_dim: int=1024):
        super(MaskedLatentMapper, self).__init__()
        self.msk = msk
        self.emb = emb

        self.condition_mapping = nn.Linear(num_conds*cond_dim, model_dim)
        self.condition_norm = nn.BatchNorm1d(model_dim)

        self.main = nn.Sequential(*[
            nn.Linear(model_dim, model_dim),
            nn.LeakyReLU(0.1),
            
            nn.Linear(model_dim, model_dim),
            nn.LeakyReLU(0.1),

            nn.Linear(model_dim, model_dim),
            nn.LeakyReLU(0.1),

            nn.Linear(model_dim, model_dim),
            nn.LeakyReLU(0.1),

            nn.Linear(model_dim, latent_dim),
            nn.Tanh()
        ])
    
    def embed_y(self, y):
        y_emb = self.emb(y)
        y_emb = self.condition_mapping(y_emb)
        y_emb = self.condition_norm(y_emb)
        return y_emb
    
    def encode(self, y):
        h = self.main(y)
        return h

    def forward(self, y):
        y_msk = self.msk(y)
        y_emb = self.embed_y(y_msk)
        h = self.encode(y_emb)
        return h


class Discriminator(nn.Module):
    '''
    The discriminator is a simple feedforward neural network.
    The goal of the discriminator is to discern latent codes generated from images, i.e. "real" samples, and latent codes generated from noise and conditons, i.e. "fake" samples
    Spectral Norm is used for superior performance of GAN architectures
    Number of channels and progression of number of neurons is taken from original publication.
    '''

    def __init__(self, latent_dim: int, model_dim: int=1024):
        super(Discriminator, self).__init__()
        assert model_dim >= 2, 'needs more than two neurons'

        """ removed from Sequential
        nn.utils.spectral_norm(nn.Linear(model_dim, model_dim)),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.2), """

        self.main = nn.Sequential(
            *[
                nn.utils.spectral_norm(nn.Linear(latent_dim, model_dim)),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.25),

                nn.utils.spectral_norm(nn.Linear(model_dim, model_dim)),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.25),

                nn.utils.spectral_norm(nn.Linear(model_dim, model_dim//2)),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.25),

                nn.utils.spectral_norm(nn.Linear(model_dim//2, 1))
            ]
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)


class mcSAALAE(nn.Module):

    def __init__(self, arch: dict, spars_params: dict):
        super(mcSAALAE, self).__init__()
        self.msk = ConditionMasking(arch, spars_params)
        self.emb = ConditionEmbedding(arch)


        self.encoder = Encoder(
            latent_dim=arch['latent_dim'],
            start_channels=arch['start_channels'],
            n_levels=arch['num_levels'],
            image_channels=arch['image_channels'],
            image_width=arch['image_width'],
            image_height=arch['image_height'],
            num_resblocks_per_level=arch['num_blocks_per_level'],
            attention_levels=arch['attention_levels'],
            attention_kwargs=arch
        )
        self.generator = Generator(
            latent_dim=arch['latent_dim'],
            noise_dim=arch['noise_dim'],
            start_channels=arch['start_channels'],
            n_levels=arch['num_levels'],
            image_height=arch['image_height'],
            image_width=arch['image_width'],
            image_channels=arch['image_channels'],
            num_resblocks_per_level=arch['num_blocks_per_level'],
            attention_levels=arch['attention_levels'],
            attention_kwargs=arch
        )
        self.discriminator = Discriminator(
            latent_dim=arch['latent_dim'],
            model_dim=arch.get('disc_dim', 1024)
        )
        self.mapper = MaskedLatentMapper(
            msk=self.msk,
            emb=self.emb,
            cond_dim=arch['y_embed_dim'],
            num_conds=len(arch['cond_dims']),
            latent_dim=arch['latent_dim'],
            model_dim=arch.get('mapper_dim', 1024)
        )

        self.noise_dim = arch['noise_dim']
    
    def generate_new(self, y: Tensor, n: Tensor=None) -> Tensor:
        if n is None:
            n = torch.randn((y.size(0), self.noise_dim)).to(y.device)
        z = self.mapper(y)
        generated = self.generator(z, n)
        return generated
    
    def reconstruct(self, x: Tensor, n: Tensor=None) -> Tensor:
        if n is None:
            n = torch.randn((y.size(0), self.noise_dim)).to(x.device)
        z = self.encoder(x)
        x_hat = self.generator(z, n)
        return x_hat
    
    def get_cycle_latent(self, y: Tensor, n: Tensor=None):
        if n is None:
            n = torch.randn((y.size(0), self.noise_dim)).to(y.device)
        generated = self.generate_new(y, n)
        z = self.encoder(generated)
        return z
    
    def get_discrimination_logits(self, z):
        return self.discriminator(z)
    
    '''
    The discriminator loss is used to update the discriminator and the encoder
    It's goal is to ensure that
    1. the discriminator learns to discern "real" latents from the encoder and "fake" latents from the mapper
    2. the encoder learns to produce latents that help the discriminator i.e. are meaningful
    '''
    def get_discriminator_loss(self, x: Tensor, y: Tensor, n: Tensor=None, gamma: float=20) -> Tensor:
        if n is None:
            n = torch.randn((y.size(0), self.noise_dim)).to(y.device)
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            x.requires_grad = True
            real_logits = self.discriminator(self.encoder(x))
            fake_logits = self.discriminator(self.encoder(self.generator(self.mapper(y), n)))
            regularization = gamma/2 * torch.mean(torch.autograd.grad(real_logits.sum(), x, create_graph=True, retain_graph=True)[0], dim=[1, 2, 3])**2

        loss = nn.functional.softplus(fake_logits) + nn.functional.softplus(-real_logits) + regularization
        return loss.mean()
    
    '''
    The generator loss is used to update the Generator and the Mapper.
    It ensures that
    1. The Mapper creates latents the generator can work with
    2. The Generator produces images from latents produced by the Mapper that fool the Discriminator when encoded by the Encoder (which however does not learn to help the Mapper/Generator for numerical stability)
    '''
    def get_generator_loss(self, y: Tensor, n: Tensor=None) -> Tensor:
        if n is None:
            n = torch.randn((y.size(0), self.noise_dim)).to(y.device)
        fake_logits = self.discriminator(self.encoder(self.generator(self.mapper(y), n)))
        loss = nn.functional.softplus(-fake_logits)
        return loss.mean()

    '''
    The latent consistency loss is used to update the Generator and the Encoder.
    It ensures that
    1. The Encoder produces latent codes that are similar to the latent codes by the Mapper
    2. The Generator produces images whos encoding is the same as the input to the Generator
    '''
    def get_latent_consistency_loss(self, y: Tensor, n: Tensor=None) -> Tensor:
        if n is None:
            n = torch.randn((y.size(0), self.noise_dim)).to(y.device)
        latent_m = self.mapper(y.clone())
        latent_cycle = self.encoder(self.generator(self.mapper(y.clone()), n))

        loss = ((latent_m.contiguous() - latent_cycle.contiguous())**2).mean()
        return loss

    # forwars is equivalent to generate new but as an ignored x input
    def forward(self, x: Tensor, y: Tensor, n: Tensor=None) -> Tensor:
        if n is None:
            n = torch.randn((y.size(0), self.noise_dim)).to(y.device)
        return self.generate_new(y.clone(), n.clone())
    
    # two methods consistency with other mc models, not necessary but allows reusing methods
    def y_emb(self, y: Tensor) -> Tensor:
        return self.mapper(y.clone())
    
    def decode(self, sample: Tensor, y: Tensor) -> Tensor:
        n = torch.randn((y.size(0), self.noise_dim)).to(y.device)
        return self.generator(y, n)
    
'''
Implements the three step training procedure from the ALAE paper.
1. Update Encoder and Discriminator based on discriminator loss
2. Update Mapper and Generator based on generator loss
3. Update Encoder and Generator based on latent consistency loss

Returns all three losses and a reconstruction loss (MSE)
'''
def update_step_mcSAALAE(model: mcSAALAE, x: Tensor, y: Tensor, optimizer_encoder: torch.optim.Optimizer, optimizer_discriminator: torch.optim.Optimizer, optimizer_mapper: torch.optim.Optimizer, optimizer_generator: torch.optim.Optimizer, n: Tensor=None, gamma: float=20.) -> tuple:
    if n is None:
        n = torch.randn((y.size(0), model.noise_dim)).to(y.device)

    model.train()

    # Step 1
    optimizer_encoder.zero_grad()
    optimizer_discriminator.zero_grad()
    optimizer_mapper.zero_grad()
    optimizer_generator.zero_grad()
    discriminator_loss = model.get_discriminator_loss(x, y.clone(), n, gamma)
    discriminator_loss.backward()
    optimizer_encoder.step()
    optimizer_discriminator.step()

    # Step 2
    optimizer_encoder.zero_grad()
    optimizer_discriminator.zero_grad()
    optimizer_mapper.zero_grad()
    optimizer_generator.zero_grad()
    generator_loss = model.get_generator_loss(y.clone(), n)
    generator_loss.backward()
    optimizer_mapper.step()
    optimizer_generator.step()

    # Step 3
    optimizer_encoder.zero_grad()
    optimizer_discriminator.zero_grad()
    optimizer_mapper.zero_grad()
    optimizer_generator.zero_grad()
    latent_consistency_loss = model.get_latent_consistency_loss(y.clone(), n)
    latent_consistency_loss.backward()
    optimizer_encoder.step()
    optimizer_generator.step()

    # Calculate MSE
    model.eval()
    with torch.no_grad():
        x_hat = model.generate_new(y.clone(), n.clone())
        mse = ((x.contiguous() - x_hat.contiguous())**2).mean()

    return discriminator_loss, generator_loss, latent_consistency_loss, mse

def get_alae_losses(model: mcSAALAE, x: Tensor, y: Tensor, n: Tensor=None, gamma: float=20.):
    if n is None:
        n = torch.randn((y.size(0), model.noise_dim)).to(y.device)

    model.eval()
    # cannot use no_grad because gradients are needed for weighting
    # Step 1
    discriminator_loss = model.get_discriminator_loss(x, y.clone(), n, gamma)


    with torch.no_grad():
        # Step 2
        generator_loss = model.get_generator_loss(y.clone(), n)
        # Step 3
        latent_consistency_loss = model.get_latent_consistency_loss(y.clone(), n)

        # Calculate MSE
        x_hat = model.generate_new(y.clone(), n)
        mse = ((x.contiguous() - x_hat.contiguous())**2).mean()

    return discriminator_loss, generator_loss, latent_consistency_loss, mse
