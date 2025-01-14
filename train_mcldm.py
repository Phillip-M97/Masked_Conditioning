import argparse
import os
import datetime
import torch
import numpy as np
import random
from omegaconf import OmegaConf
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from diffusers import AutoencoderKL

from modules.mc_ldm import mcLDM
from modules.data import get_data_module
from modules.sparsity_scheduler import LinearSpars, ConstantSpars, StepSpars, ExponentialSpars, CosineAnnealingSpars

def seed_everything(seed):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_args() -> dict:
    parser = argparse.ArgumentParser("mcDiffusion")
    parser.add_argument('--cfg_name', type=str, required=True)
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--use_es', action='store_true')
    parser.add_argument('--run_id', type=str, default=None, required=False)
    parser.add_argument('--ckpt_path', type=str, default=None)
    args = parser.parse_args()
    return args

def load_cfg(cfg_name: str) -> tuple:
    cfg = OmegaConf.load(os.path.join('configs/', cfg_name + '.yaml'))
    eps_model_arch = OmegaConf.to_container(cfg['eps_model_arch'], resolve=True)
    diffusion_cfg = OmegaConf.to_container(cfg['diffusion_cfg'], resolve=True)
    params = OmegaConf.to_container(cfg['params'], resolve=True)
    spars_params = OmegaConf.to_container(cfg['spars_params'], resolve=True)

    if spars_params['sparsity_scheduler'] == 'linear':
        spars_params['sparsity_scheduler'] = LinearSpars
    elif spars_params['sparsity_scheduler'] == 'constant':
        spars_params['sparsity_scheduler'] = ConstantSpars
    elif spars_params['sparsity_scheduler'] == 'step':
        spars_params['sparsity_scheduler'] = StepSpars
    elif spars_params['sparsity_scheduler'] == 'exponential':
        spars_params['sparsity_scheduler'] = ExponentialSpars
    elif spars_params['sparsity_scheduler'] == 'cosine':
        spars_params['sparsity_scheduler'] = CosineAnnealingSpars
    else:
        raise NotImplementedError(f"Sparsity Schedule {spars_params['sparsity_scheduler']} is not known, choose linear, constant, step, exponential or cosine")
    
    if eps_model_arch['nonlinearity'] == 'selu':
        eps_model_arch['nonlinearity'] = torch.nn.SELU()
    elif eps_model_arch['nonlinearity'] == 'silu':
        eps_model_arch['nonlinearity'] = torch.nn.SiLU()
    elif eps_model_arch['nonlinearity'] == 'relu':
        eps_model_arch['nonlinearity'] = torch.nn.ReLU()
    elif eps_model_arch['nonlinearity'] == 'leaky_relu':
        eps_model_arch['nonlinearity'] = torch.nn.LeakyReLU()
    elif eps_model_arch['nonlinearity'] == 'gelu':
        eps_model_arch['nonlinearity'] = torch.nn.GELU()
    else:
        raise NotImplementedError(f"Activation function {eps_model_arch['nonlinearity']} is not known choose selu, sili, relu, leaky_relu, or gelu")

    params['epochs'] = params['trainer_params']['max_epochs']
    params['steps'] = params['trainer_params']['max_steps']
    spars_params['use_epochs'] = True if params['epochs'] != -1 else False
    return eps_model_arch, diffusion_cfg, params, spars_params

def main(args: dict, eps_model_arch: dict, diffusion_cfg: dict, params: dict, spars_params: dict):
    seed_everything(2024)

    """ Setup Logging """
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_name = f"{args.name}_{now}" if args.name is not None else now
    log_dir = f'./logs/{exp_name}'
    ckpt_dir = log_dir + '/checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)

    csv_logger = CSVLogger(
        log_dir,
        name=exp_name,
        version="",
        prefix="",
        flush_logs_every_n_steps=500
    )
    hyperparams = {'args': args, 'eps_model_arch': eps_model_arch, 'diffusion_cfg': diffusion_cfg, 'params': params, 'spars_params': spars_params}
    if not args.ckpt_path:
        csv_logger.log_hyperparams(hyperparams)
    wandb_logger = WandbLogger(
        dir=log_dir,
        name=exp_name,
        project='mcldm',
        config=hyperparams,
        mode='online' if args.use_wandb else 'disabled',
        resume='allow',
        id=args.run_id
    )
    logger = {wandb_logger, csv_logger}

    """ Set up Model """
    vae = AutoencoderKL.from_pretrained(eps_model_arch['vae_model_id'], subfolder='vae', use_safetensors=True)  # torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    model = mcLDM(
        eps_model_arch=eps_model_arch,
        vae=vae,
        diffusion_cfg=diffusion_cfg,
        params=params,
        spars_params=spars_params,
        calculate_fid=args.use_es,
        use_precomputed_latents=True if 'latent' in params['data_module'] else False,
        accumulate_grad_batches=params['trainer_params']['accumulate_grad_batches']
    )

    """ Set up Data """
    data = get_data_module(
        module_type=params['data_module'],
        images_dir=params['images_dir'],
        conditions_df_path=params['conditions_csv'],
        vae=vae,
        device='cuda' if torch.cuda.is_available else 'cpu',
        batch_size=params['batch_size'],
        image_size=params['image_size'],
        train_split=params['train_split']
    )

    """ Set up Callbacks """
    ckpt_params = params["checkpoint_callback_params"]
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="step{step:06d}",
        monitor=ckpt_params['monitor'],
        mode=ckpt_params['mode'],
        every_n_train_steps=ckpt_params['every_n_train_steps'],
        save_last=ckpt_params['save_last'],
        save_top_k=ckpt_params['save_top_k'],
        auto_insert_metric_name=ckpt_params['auto_insert_metric_name']
    )
    # early stopping based on FID calculated from 100 samples, inspired by SA-ALAE paper
    es_callback = EarlyStopping(
        monitor='val/fid',
        min_delta=0.0,
        patience=3*10,
        verbose=False,
        mode='min'
    )

    """ Setup Trainer """
    if torch.cuda.is_available():
        print('Using GPU')
        gpu_kwargs = {
            'devices': len([torch.cuda.device(i) for i in range(torch.cuda.device_count())]),
            'accelerator': 'gpu',
        }
        if gpu_kwargs['devices'] > 1:
            gpu_kwargs['strategy'] = 'ddp_find_unused_parameters_true'
    else:
        print('Using CPU')
        gpu_kwargs = {'accelerator': 'cpu'}

    trainer_params = params['trainer_params']
    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, es_callback] if args.use_es else [checkpoint_callback],
        max_epochs=trainer_params['max_epochs'],
        max_steps=trainer_params['max_steps'],
        num_sanity_val_steps=trainer_params['num_sanity_val_steps'],
        accumulate_grad_batches=trainer_params['accumulate_grad_batches'],
        limit_val_batches=trainer_params['limit_val_batches'],
        precision=trainer_params['precision'],
        log_every_n_steps=trainer_params['log_every_n_steps'],
        devices=gpu_kwargs.get('devices', 'auto'),
        accelerator=gpu_kwargs.get('accelerator', 'auto'),
        strategy=gpu_kwargs.get('strategy', 'auto')
    )

    """ Train """
    trainer.fit(model, data, ckpt_path=args.ckpt_path)
    print('\n\nYour Model is trained 🥳')

if __name__ == '__main__':
    args  = parse_args()
    eps_model_arch, diffusion_cfg, params, spars_params = load_cfg(args.cfg_name)
    main(args, eps_model_arch, diffusion_cfg, params, spars_params)
