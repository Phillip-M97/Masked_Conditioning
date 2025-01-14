import os
import torch
import pandas as pd
from torch.utils.data import random_split, DataLoader
import random
import argparse
from datetime import datetime
import json
from copy import deepcopy
from tabulate import tabulate
import numpy as np

from modules.data import BikeSketchDataset
from modules.sparsity_scheduler import LinearSpars
from modules import conv_mcvae
from modules.globals import DEVICE
from modules import utils

SEED = 42
TRAIN_SPLIT = 0.80
BATCH_SIZE = 140

torch.manual_seed(SEED)
random.seed(SEED)

def get_configuration_biked(df_Y: pd.DataFrame):
    arch = {
        'image_channels':       1,
        'im_embed_dim':         16,
        'start_channels':       16,
        'num_levels':           5,      # 5 resolution halvings to for our 256x256 images we end up with 8x8 latents
        'num_blocks_per_level': 2,
        'nonlinearity':         torch.nn.SELU(),
        'dropout':              0.15,
        'latent_dim':           4,      # note that the latent dim (aka number of channels in the latent) is smaller than the number of halvings s.t. compression is necessary
        'cond_dims':            ['bs_dim', 'tc_dim', 'bst_dim', 'bdt_dim', 'fs_dim', 'rsf_dim', 'rsr_dim', 'ft_dim'],
        'numerical_cond':       ['tc_dim'],
        'bs_dim':               df_Y['BikeStyle'].nunique(),
        'tc_dim':               1,
        'bst_dim':              2,
        'bdt_dim':              2,
        'fs_dim':               df_Y['frame_size'].nunique(),
        'rsf_dim':              df_Y['RIM_STYLE front'].nunique(),
        'rsr_dim':              df_Y['RIM_STYLE rear'].nunique(),
        'ft_dim':               df_Y['Fork type'].nunique(),
        'y_embed_dim':          10,
        'attention_levels':     [3, 4],  # attention on second to last (32x32) and last (16x16) level
        'attention_heads':      3,
        'attention_head_dim':   64
    }

    params = {
        'batch_size':           140,
        'epochs':               400,
        'lr':                   0.0018,
        'betas':                (.54, .94),
        'kld_weight':           0.9,
        'weight_decay':         0.15,
        'mean_reduce':          True
    }

    spars_params = {
        'sparsity_scheduler':   LinearSpars,
        'sparsity':             0.1,
        'step_size':            100,
        'target_sparsity':      0.2,
        'target_epoch':         -1,
        'mask_value':           -1
    }
    return arch, params, spars_params

def parse_args() -> dict:
    parser = argparse.ArgumentParser('Conv-mcVAE')
    parser.add_argument('--images_path', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--name', type=str, default='conv-mcVAE-Biked')
    parser.add_argument('--type', type=str, default='biked')
    return parser.parse_args()

def main():
    args = parse_args()

    df_Y = pd.read_csv(args.csv_path)
    if args.type == 'biked':
        data = BikeSketchDataset(args.images_path, df_Y, keep_cpu=False)
        arch, params, spars_params = get_configuration_biked(df_Y)
    else:
        raise NotImplementedError('Currently only "biked" is allowed as type')

    train_size = int(TRAIN_SPLIT * len(data))
    test_size = len(data) - train_size
    train_data, test_data = random_split(data, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    print('\n-- Training--')
    model = conv_mcvae.ConvMCVae(arch, spars_params).to(DEVICE)
    utils.train(model, train_loader, test_loader, 2, params, spars_params)
    print('--Finished Training--')

    print('\n-- Saving Model --')
    # save model
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = f'./logs/{args.name}-{now}'
    os.makedirs(path, exist_ok=True)
    fname = 'mcVAE-weights.pt'
    torch.save(model.state_dict(), os.path.join(path, fname))
    # save hyperparameters
    fname = 'hparams.json'
    hparams = {'arch': deepcopy(arch), 'params': deepcopy(params), 'spars_params': deepcopy(spars_params)}
    hparams['arch']['nonlinearity'] = arch['nonlinearity'].__class__.__name__
    hparams['spars_params']['sparsity_scheduler'] = spars_params['sparsity_scheduler'].__class__.__name__
    with open(os.path.join(path, fname), 'w') as f:
        json.dump(hparams, f)
    print('-- Model Saved --')

    print('\n-- Evaluation --')
    mse_per_sparsity = utils.img_evaluate_for_all_spars(model, test_loader, spars_params, arch)
    mean = sum(mse_per_sparsity)/len(mse_per_sparsity)
    headings = ['Sparsity >'] + [str(s) for s in np.linspace(0, 0.9, 10, dtype=np.float16)] + ['Mean']
    print('MSE Loss per Sparsity',
        tabulate([['Conv mcVAE'] + mse_per_sparsity + [mean]], headers=headings, tablefmt='mixed_grid', numalign='right'),
        sep='\n'
    )

    print('\n\nYour Model is trained 🥳')


if __name__ == '__name__':
    main()