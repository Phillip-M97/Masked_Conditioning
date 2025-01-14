import torch
from torch import nn
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from diffusers import AutoencoderKL
import einops
import matplotlib.pyplot as plt
import wandb
from math import ceil
from PIL import Image as PILImage
from datetime import datetime
import numpy as np

from .conv_mcvae import DownBlock, UpBlock, ResBlock, SpatialSelfAttention
from .emb_mcvae import ConditionEmbedding, ConditionMasking
from modules.diffusion_components.unet import mcUnet
from modules.diffusion_components.beta_schedule import BetaSchedule

"""
This implmentation is inspired by and partially copied from
    - PlantLDM https://github.com/joh-fischer/PlantLDM
    - Latent Diffusion https://github.com/CompVis/latent-diffusion
"""

class mcLDM(LightningModule):

    def __init__(
            self,
            eps_model_arch: dict,
            vae: AutoencoderKL,
            diffusion_cfg: dict,
            params: dict,
            spars_params: dict,
            calculate_fid: bool=False,
            use_precomputed_latents: bool=False,
            accumulate_grad_batches: int=1
    ) -> None:
        super().__init__()

        self.n_steps = diffusion_cfg['n_steps']
        self.vae_model = vae
        self.lr = params['lr']
        self.wd = params['weight_decay']
        self.optim_betas = (params['beta_1'], params['beta_2'])
        self.validation_samples = None
        self.use_precomputed_latents = use_precomputed_latents
        self.accumulate_grad_batches = accumulate_grad_batches

        self.calculate_fid = calculate_fid
        self.c_fid = 1000
        if self.calculate_fid:
            self.fid = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize=True)
            self.num_val_samples = 100
        else:
            self.fid = None
            self.num_val_samples = 8

        self.msk = ConditionMasking(arch=eps_model_arch, spars_params=spars_params)
        self.y_emb = ConditionEmbedding(arch=eps_model_arch)

        self.spars_use_epochs = spars_params.get('use_epochs', True)
        self.sparsity_scheduler = spars_params['sparsity_scheduler'](self, eps_model_arch, params, spars_params, use_epochs=self.spars_use_epochs)

        # define diffusion backbone model, here a time-conditional, masked-conditional Unet (mainly inspired by the conv_mcvae architecture)
        self.eps_model = mcUnet(
            # image_channels=3,  # use this for non latent diffusion
            image_channels=self.vae_model.latent_channels,
            model_channels=eps_model_arch['start_channels'],
            num_levels=eps_model_arch['num_levels'],
            num_resblocks_per_level=eps_model_arch['num_blocks_per_level'],
            nonlinearity=eps_model_arch['nonlinearity'],
            time_emb_dim=eps_model_arch['time_emb_dim'],
            pos_emb_dim=eps_model_arch['pos_emb_dim'],
            msk=self.msk,
            y_emb=self.y_emb,
            num_conditions=len(eps_model_arch['cond_dims']),
            condition_dim=eps_model_arch['y_embed_dim'],
            dropout=eps_model_arch['dropout'],
            attention_levels=eps_model_arch['attention_levels'],
            num_attention_heads=eps_model_arch['attention_heads'],
            attention_dim=eps_model_arch['attention_head_dim'],
        )

        # freeze vae model
        for param in self.vae_model.parameters():
            param.requires_grad = False

        # check validity of parameters
        if not diffusion_cfg['beta_1'] < diffusion_cfg['beta_2'] < 1.0:
            raise ValueError(f"beta1: {diffusion_cfg['beta_1']} < beta2: {diffusion_cfg['beta_2']} < 1.0 not fulfilled")
        available_beta_schedules = ["linear", "quadratic", "sigmoid", "cosine"]
        if diffusion_cfg['beta_schedule'] not in available_beta_schedules:
            raise ValueError(f"Beta schedule should be one of the following: {available_beta_schedules}")
        available_loss_functions = ["l1", "l2", "huber"]
        if params['loss_fn'] not in available_loss_functions:
            raise ValueError(f"Loss function should be one of the following: {available_loss_functions}")
        self.loss_function = params['loss_fn']

        # setup beta schedule
        self.beta_1 = diffusion_cfg['beta_1']
        self.beta_2 = diffusion_cfg['beta_2']
        self.betas = BetaSchedule(self.beta_1, self.beta_2, diffusion_cfg['beta_schedule'], self.n_steps).values

        # define alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_precomputed_latents:
            return x
        # return x # uncomment to use non latent diffusion model
        # x should be normalized to [-1, 1] range
        self.vae_model.eval()
        latent = self.vae_model.encode(x.clone())
        z = latent.latent_dist.sample()
        # print('vae z sample nan: ', torch.isnan(z).any().item())
        return z * 0.18215  # the normalization value 0.18215 is used by the SD authors to ensure unit variance in the latent space (the KL-weight of the first stage VAE was very small), I have no idea how this number was determined though and is simply hard-coded by them for all SD first stages
    
    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        # return x # uncomment to run non latent diffusion model
        self.vae_model.eval()
        x = (1 / 0.18215) * x
        img_t = self.vae_model.decode(x).sample
        # img_t is in [-1, 1] range
        return img_t
    
    def extract(self, a, t, x_shape):
        """
        extracts an appropriate t index for a batch of indices
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    # noising procedure from start image according to diffusion algorithm
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    # use mcUnet to predict current noise based on noisy image, conditioning information, and timestep
    def p_predict(self, x_noisy: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        predicted_noise = self.eps_model(x_noisy, y, t)
        return predicted_noise

    # single forward diffusion step
    def diffusion(self, x_start: torch.Tensor, y: torch.Tensor, noise: torch.Tensor=None):
        # pass x through first stage encoder
        x_start = self.encode(x_start.clone())

        # ensure same device and sample noise from normal distribution is necessary
        if noise is None:
            noise = torch.randn_like(x_start)
        if noise.device != x_start.device:
            noise = noise.to(x_start.device)
        if self.sqrt_alphas_cumprod.device != x_start.device:
            self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(x_start.device)  
        if self.sqrt_one_minus_alphas_cumprod.device != x_start.device:
            self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(x_start.device)

        # sample random timestep
        t = torch.randint(0, self.n_steps, (x_start.shape[0],))
        if t.device != x_start.device:
            t = t.to(x_start.device)
        
        # noise x
        x_noisy = self.q_sample(x_start.clone(), t, noise)

        # predict noise based on noisy x, condition information and timestep
        predicted_noise = self.p_predict(x_noisy, y, t)
        return noise, predicted_noise
    
    # simplified diffusion loss is L1, L2 or Huber loss between noise and predicted noise
    def diffusion_loss(self, x_start: torch.Tensor, y: torch.Tensor, noise: torch.Tensor=None) -> torch.Tensor:
        noise, predicted_noise = self.diffusion(x_start, y, noise)
        return self.calculate_loss(noise, predicted_noise)
    
    def calculate_loss(self, noise, predicted_noise):
        if self.loss_function == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        elif self.loss_function == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif self.loss_function == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError(f'The loss function "{self.loss_function}" is not (yet) implemented')
        return loss
    
    """ Sampling Functions """

    # p_sample is used to get the image at timestep t_index provided x at timestep t_index+1
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, t_index: int, cfg_scale: float=0., mask_y: bool=True) -> torch.Tensor:
        # ensure tensors are on same device
        if t.device != x.device:
            t = t.to(x.device)
        if self.betas.device != x.device:
            self.betas = self.betas.to(x.device)
        if self.sqrt_recip_alphas.device != x.device:
            self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(x.device)
        if self.sqrt_one_minus_alphas_cumprod.device != x.device:
            self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(x.device)
        if self.posterior_variance.device != x.device:
            self.posterior_variance = self.posterior_variance.to(x.device)

        # get constant beta, alpha and 1/alpha for current timestep
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        if cfg_scale <= 0.:
            pred_noise = self.eps_model(x, y, t, mask_y=mask_y)
        else:
            previous_p = self.msk.p
            previous_training = self.msk.training
            # predict unconditional noise by setting masking probability to 1
            self.msk.p = 1.0
            self.msk.training = True  # masking is only applied in training mode
            pred_noise_uncond = self.eps_model(x, y, t, mask_y=mask_y)
            # predict conditional noise by setting masking probability back to current value
            self.msk.p = previous_p
            self.msk.training = previous_training
            pred_noise_cond = self.eps_model(x, y, t, mask_y=mask_y)
            # calculate cfg noise according to equation 6 from https://arxiv.org/pdf/2207.12598
            pred_noise = (1 + cfg_scale) * pred_noise_cond - cfg_scale * pred_noise_uncond

        # calculate predicted denoised image
        x_0 = sqrt_recip_alphas_t * (x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            # if we are at timestep 0 the current predicted image is the final prediction of the model
            return x_0
        else:
            # if we are at a timestep != 0 we add noise to the prediction to get the image at timestep t (if x is at timestep t+1)
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)  # variance
            # reparametrization trick
            noise = torch.randn_like(x)  # get standard gaussian noise
            if noise.device != x.device:
                noise = noise.to(x.device)
            x_t = x_0 + torch.sqrt(posterior_variance_t) * noise
            return x_t
    
    # p_sample_loop iterates from n_steps to 0 and iteratively denoises initial gaussian noise to generate an image given the condition vector y
    @torch.no_grad()
    def p_sample_loop(self, shape: tuple, y: torch.Tensor, cfg_scale: float=0., sample_steps=None, mask_y: bool=True):
        assert shape[0] == y.size(0), 'Batch dimension of shape and batch dimension of y must match, we need exactly one condition per image'
        assert cfg_scale >= 0., 'Sampling requires cfg_scale >= 0'

        img = torch.randn(shape)  # get initial noise (!! dimensions are in latent, 4 channels, img width//8, img heigh//8)
        if img.device != y.device:
            img = img.to(y.device)
        
        imgs = []
        for timestep in reversed(range(0, self.n_steps)):
            # current image is calculated using p_sample
            t = torch.full((shape[0],), timestep, dtype=torch.long)
            img = self.p_sample(img, y, t, timestep, cfg_scale=cfg_scale, mask_y=mask_y)
            # save noisy images at certain timesteps
            if sample_steps is None or (sample_steps is not None and timestep in sample_steps):
                imgs.append(img)
        return imgs

    @torch.no_grad()
    def p_sample_ddim(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, next_t: torch.Tensor, t_index: int, cfg_scale: float=0., eta=0.) -> torch.Tensor:
        # ensure tensors are on same device
        if t.device != x.device:
            t = t.to(x.device)
        if self.alphas_cumprod.device != x.device:
            self.alphas_cumprod = self.alphas_cumprod.to(x.device)

        # get constant alpha cumprod for current and next timestep
        alpha_cumprod_t = self.extract(self.alphas_cumprod, t, x.shape)
        next_alpha_cumprod_t = self.extract(self.alphas_cumprod, next_t, x.shape) # if next_t is not None else None

        if cfg_scale <= 0.:
            pred_noise = self.eps_model(x, y, t)
        else:
            # TODO: it might be more efficient to generate conditional and unconditional samples in parallel using batching
            previous_p = self.msk.p
            previous_training = self.msk.training
            # predict unconditional noise by setting masking probability to 1
            self.msk.p = 1.0
            self.msk.training = True
            pred_noise_uncond = self.eps_model(x, y, t)
            # predict conditional noise by setting masking probability back to current value
            self.msk.p = previous_p
            self.msk.training = previous_training
            pred_noise_cond = self.eps_model(x, y, t)
            # calculate cfg noise according to equation 6 from https://arxiv.org/pdf/2207.12598
            pred_noise = (1 + cfg_scale) * pred_noise_cond - cfg_scale * pred_noise_uncond 

        x_0 = (x - pred_noise * torch.sqrt(1-alpha_cumprod_t)) / torch.sqrt(alpha_cumprod_t)
        
        sigma = eta * torch.sqrt((1-next_alpha_cumprod_t)/(1-alpha_cumprod_t) * (1 - alpha_cumprod_t/next_alpha_cumprod_t))
        x_t_direction = torch.sqrt(1 - next_alpha_cumprod_t - sigma**2) * pred_noise

        noise = torch.randn_like(x)
        x_t_minus_one = torch.sqrt(next_alpha_cumprod_t) * x_0 + sigma * noise + x_t_direction
        return x_t_minus_one
    
    def get_ddim_timesteps(self, num_steps: int) -> list:
        # Create a list of timesteps
        assert num_steps > 0 and num_steps <= self.n_steps, 'num_steps must be between 1 and T'
        timesteps = [int(round(i)) for i in np.linspace(0, self.n_steps-1, num_steps)]
        timesteps.reverse()
        return timesteps

    @torch.no_grad()
    def p_sample_loop_ddim(self, shape: tuple, y: torch.Tensor, cfg_scale: float=0., num_steps: int=50, eta: float=0., sample_steps: list=None) -> list:
        assert shape[0] == y.size(0), 'Batch dimension of shape and batch dimension of y must match, we need exactly one condition per image'
        assert cfg_scale >= 0., 'Sampling requires cfg_scale >= 0'

        # Initialize image with noise
        img = torch.randn(shape)
        if img.device != y.device:
            img = img.to(y.device)

        imgs = []
        timesteps = self.get_ddim_timesteps(num_steps)
        prev_timesteps = timesteps[1:]
        timesteps = timesteps[:-1]
        for i, timestep in enumerate(timesteps):
            # calculate current image using ddim sampling
            t = torch.full((shape[0],), timestep, dtype=torch.long, device=img.device)
            next_t = torch.full((shape[0],), prev_timesteps[i], dtype=torch.long, device=img.device) # if i < len(timesteps)-1 else None
            img = self.p_sample_ddim(img, y, t, next_t, timestep, cfg_scale, eta)
            imgs.append(img)
        return imgs

    @torch.no_grad()
    def sample(self, image_size: tuple, y: torch.Tensor, cfg_scale: float=0.0, sample_steps=None) -> torch.Tensor:
        shape = (y.size(0), self.vae_model.latent_channels, image_size[0]//8, image_size[1]//8)
        # shape = (y.size(0), 3, image_size[0], image_size[1])  # use this for non latent model
        latent_sample = self.p_sample_loop(shape, y, cfg_scale=cfg_scale, sample_steps=sample_steps)
        return self.decode(latent_sample[-1].detach())

    @torch.no_grad()
    def sample_with_masked_y(self, image_size: tuple, y_msk: torch.Tensor, cfg_scale: float=0.0, sample_steps=None) -> torch.Tensor:
        shape = (y_msk.size(0), self.vae_model.latent_channels, image_size[0]//8, image_size[1]//8)
        # shape = (y.size(0), 3, image_size[0], image_size[1])  # use this for non latent model
        latent_sample = self.p_sample_loop(shape, y_msk, cfg_scale=cfg_scale, sample_steps=sample_steps, mask_y=False)
        return self.decode(latent_sample[-1].detach())
    
    @torch.no_grad()
    def ddim_sampling(self, image_size: tuple, y: torch.Tensor, cfg_scale: float=0.0, num_steps: int=50, eta: float=0., sample_steps=None) -> torch.Tensor:
        shape = (y.size(0), self.vae_model.latent_channels, image_size[0]//8, image_size[1]//8)
        # shape = (y.size(0), 3, image_size[0], image_size[1])  # use this for non latent model
        latent_sample = self.p_sample_loop_ddim(shape, y, cfg_scale=cfg_scale, num_steps=num_steps, eta=eta, sample_steps=sample_steps)
        return self.decode(latent_sample[-1].detach())
    
    """ PyTorch Lightning """

    # for each training step we simply calculate the current diffusion loss and backpropagate it
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        losses = self.diffusion_loss(x, y)
        # print(losses)
        if not self.spars_use_epochs:
            self.sparsity_scheduler.step()
            self.log('sparsity', self.sparsity_scheduler.get_last_sparsity())
        self.log('train/loss', losses.detach().cpu().item(), batch_size=x.size(0))
        return losses
    
    def on_train_epoch_end(self):
        # step sparsity scheduler at the end of each epoch
        if self.spars_use_epochs:
            if self.global_step % self.accumulate_grad_batches == 0:
                self.sparsity_scheduler.step()
            self.log('sparsity', self.sparsity_scheduler.get_last_sparsity())
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        losses = self.diffusion_loss(x, y).detach().cpu().item()
        # print(losses)
        if self.validation_samples is None:
            self.validation_samples = batch
        if self.validation_samples[0].size(0) < self.num_val_samples:
            self.validation_samples[0] = torch.cat((self.validation_samples[0], x), dim=0)
            self.validation_samples[1] = torch.cat((self.validation_samples[1], y), dim=0)
        if self.validation_samples[0].size(0) > self.num_val_samples:
            self.validation_samples[0] = self.validation_samples[0][:self.num_val_samples]
            self.validation_samples[1] = self.validation_samples[1][:self.num_val_samples]
        self.log('val/loss', losses, batch_size=x.size(0))
        return losses
    
    # at the end of each epoch we sample an image for a 8 examples from the validation data and send it to wandb if wandb is configures
    def on_validation_epoch_end(self) -> None:
        if self.validation_samples is None:
            return
        x, y = self.validation_samples
        if (self.current_epoch+1) % 10 == 0:
            # if we do not calculate the fid we only need 8 samples for the images
            if not self.calculate_fid:
                x, y = x[:min(8, self.num_val_samples)], y[:min(8, self.num_val_samples)]
            samples = []
            # batched sampling with batchsize 16
            for i in range(ceil(x.size(0) / 16)):
                c_x, c_y = x[i*16:min(x.size(0), (i+1)*16)], y[i*16:min(y.size(0), (i+1)*16)]
                if self.use_precomputed_latents:
                    # sample expects final image size so we need to calculate the size of the real image from the size of the latent
                    c_sample = self.sample((c_x.size(2)*8, c_x.size(3)*8), c_y)
                else:
                    c_sample = self.sample((c_x.size(2), c_x.size(3)), c_y)
                samples.append(c_sample)

            sample = torch.cat(samples, dim=0)
            if self.use_precomputed_latents:
                x = self.decode(x)  # if x is a latent we have to decode it to an image first
            x_de, sample_de = (x/2 + 0.5).clamp(0, 1), (sample/2 + 0.5).clamp(0, 1)
        
            if self.calculate_fid:
                # FID model expects inputs of shape (bs, 3, 299, 299)
                x_fid = x_de.repeat((1, 3, 1, 1)) if x_de.size(1) == 1 else x_de
                sample_fid = sample_de.repeat((1, 3, 1, 1)) if sample_de.size(1) == 1 else sample_de
                resize_op = transforms.Resize((299, 299))
                x_fid = resize_op(x_fid)
                sample_fid = resize_op(sample_fid)
                self.fid.reset()
                self.fid.update(x_fid, real=True)
                self.fid.update(sample_fid, real=False)
                c_fid = self.fid.compute()
                self.c_fid = c_fid

            # we only log 8 samples
            x_de, sample_de = x_de[:min(8, self.num_val_samples)], sample_de[:min(8, self.num_val_samples)]
            x_de, sample_de = x_de.detach().cpu(), sample_de.detach().cpu()
            # convert [0, 1] floats to [0, 255] ints
            x_de, sample_de = (x_de * 255).to(torch.uint8), (sample_de * 255).to(torch.uint8)
            # append sampled images to original images along height dimension
            combined = torch.stack(
                [torch.cat((orig, recon), dim=1) for orig, recon in zip(x_de, sample_de)]
                , dim=0
            )
            # concatenate samples along width dimension and move channel dimension to end
            combined = einops.rearrange(combined, "b c h w -> h (b w) c")
            # create RGB and BW image
            combined = combined.detach().cpu().numpy()
            image = PILImage.fromarray(combined, mode='RGB')
            image_color = wandb.Image(image)
            image_bw = image.convert('L')
            image_bw = wandb.Image(image_bw, mode='L')
            # log image to wandb
            for l in self.loggers:
                if isinstance(l, WandbLogger):
                    l.log_image('val/samples', [image_color, image_bw])

        # log FID every epoch if needed
        if self.calculate_fid:
            self.log('val/fid', self.c_fid, batch_size=x.size(0))
        return super().on_validation_epoch_end()
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd, betas=self.optim_betas)
        return optimizer
