import torch
import torch.nn as nn
import os
import json
from copy import deepcopy
from datetime import datetime
from modules.utils import preprocess_biked_data
from sklearn.model_selection import train_test_split
from modules.utils import RefPointData
from torch.utils.data import DataLoader
from modules.sparsity_scheduler import LinearSpars
import torch
from modules.emb_mcvae import mcVAE
from modules.utils import train, plot_loss, evaluate_inference
import numpy as np
from tabulate import tabulate
import pandas as pd
import argparse
from modules.globals import DEVICE
from modules.emb_mcvae import ConditionMasking
from modules.utils import sample

SEED = 42
TRAIN_SPLIT = 0.85
BATCH_SIZE = 128
EPOCHS = 500

# this function ensures that all conditions are present in the training data
def ensure_good_train_test_split(X, Y, arch):
    i = 0
    while True:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=TRAIN_SPLIT, shuffle=True, random_state=SEED+i)
        missings = []
        for j, cond in enumerate(arch['cond_dims']):
            if cond in arch['numerical_cond']:
                missings.append(0)
            else:
                num_unique = np.unique(Y_train[:, j]).shape[0]
                missings.append(arch[cond] - num_unique)
        if sum(missings) == 0:
            print('Seed is: ', SEED+i)
            break
        i += 1
    
    train_set = RefPointData(X_train, Y_train)
    test_set = RefPointData(X_test, Y_test)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    return X_train, X_test, Y_train, Y_test, train_set, test_set, train_loader, test_loader, SEED+i

# this function returns the appropriate model configuration
def get_architecture_biked(df_conditions: pd.DataFrame, X: np.array):
    arch = {
        'rp_dim': X.shape[1],                           # ref points input dimension
        'cond_dims': ['bs_dim', 'tc_dim', 'bst_dim', 'bdt_dim', 'fs_dim', 'rsf_dim', 'rsr_dim', 'ft_dim'],
        'bs_dim': len([x for x in df_conditions.columns if 'BikeStyle' in x]),
        'tc_dim': len([x for x in df_conditions.columns if 'TeethChain' in x]),
        'bst_dim': 2,
        'bdt_dim': 2,
        'fs_dim': len([x for x in df_conditions.columns if 'FrameSize' in x]),
        'rsf_dim': len([x for x in df_conditions.columns if 'RimStyleFront' in x]),
        'rsr_dim': len([x for x in df_conditions.columns if 'RimStyleRear' in x]),
        'ft_dim': len([x for x in df_conditions.columns if 'ForkType' in x]),
        'dc_enc_dim': 6,                        # positional encoding for more impact of the drag coeff
        'y_embed_dim': 5,                       # dimensionality of embeddings for embedding mcVAE
        'numerical_cond': ['tc_dim'],           # which conditions are numerical for embedding
        'rp_embed_dim': 64,                     # embedding layer for reference points
        'cond_embed_dim': 10,                   # embedding layer for condition
        'hidden_dim': [128, 64, 32],            # encoder/decoder layers
        'latent_dim': 10,                       # latent, mean, logvar dimension
        'activation': nn.SELU(),                # activation function
    }

    params = {
        'batch_size': BATCH_SIZE,                       # batch size
        'epochs': EPOCHS,                         # number of epochs
        'lr': 0.001,                             # learning rate for adam optimizer
        'betas': (.9, .99),                    # default betas for adam optimizer
        'kld_weight': 0.7,                     # kld weight - similar to beta-VAE
        'weight_decay': 0.1,                   # weight decay for adam optimizer
    }

    spars_params = {
        'sparsity_scheduler': LinearSpars,             # scheduler type
        'sparsity': 0.1,                         # initial sparsity
        'step_size': 100,                      # step size for StepSparsity only
        'target_sparsity': 0.2,                  # sparsity that shall be reached at sparsity_epoch
        'target_epoch': -1,                     # epoch where target_sparsity shall be reached
        'mask_value': -1
    }

    return arch, params, spars_params

def get_Y_per_spars(Y: np.array, spars_params: dict, arch: dict, min: float=0.0, max: float=0.9, step: float=0.1):
    c_spars_params = spars_params.copy()
    c_spars_params['sparsity'] = 0.
    c_spars_params['target_sparsity'] = 1.

    arch['bst_dim'] = 2
    arch['bdt_dim'] = 2
    msk = ConditionMasking(arch, c_spars_params)
    
    y_per_spars = []
    for c_spars in np.linspace(min, max, int((max-min)/step)+1):
        msk.p = c_spars
        y_per_spars.append(msk(torch.tensor(Y, dtype=torch.float, device=DEVICE)))
    return y_per_spars

# this function is used to calculate the MSE of the model on different sparsity settings
def evaluate_for_all_spars(model: nn.Module, X: np.array, Y: np.array, spars_params: dict, arch: dict, min: float=0.0, max: float=0.9, step: float=0.1):
    y_per_spars = get_Y_per_spars(Y, spars_params, arch, min, max, step)
    x = torch.tensor(X, dtype=torch.float).to(DEVICE)
    z_sample = torch.concat([sample(arch) for _ in range(len(Y))], dim=0).to(DEVICE)

    model = model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        mse_per_sparsity = []
        for i, c_y in enumerate(y_per_spars):
            c_y_emb = model.y_emb(c_y.to(DEVICE))
            X_hat = model.decode(z_sample, c_y_emb)
            c_mse = nn.functional.mse_loss(X_hat, x, reduction='sum')/len(c_y)
            mse_per_sparsity.append(c_mse.detach().cpu().round(decimals=3).item())
    return mse_per_sparsity

def parse_args() -> dict:
    parser = argparse.ArgumentParser('mcVAE')
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--name', type=str, default='mcVAE-Biked')
    parser.add_argument('--type', type=str, default='biked')
    return parser.parse_args


def main():
    args = parse_args()

    if args.type == ' biked':
        points, conditions_old, conditions_new, conditions_unchanged, normalization = preprocess_biked_data(args.csv_path, return_categorical=True)
        X = points.to_numpy(dtype=float)
        Y = conditions_new.to_numpy(dtype=float)
        arch, params, spars_params = get_architecture_biked(conditions_old, X)
    else:
        raise NotImplementedError('Currently only "biked" is allowed as type')

    _, X_test, _, Y_test, _, _, train_loader, test_loader, seed = ensure_good_train_test_split(X, Y, arch)
    print('Seed: ', seed)
    
    # training
    print('\n-- Training --')
    model = mcVAE(arch, spars_params).to(DEVICE)
    train(model, train_loader, test_loader, 2, params, spars_params)
    print('-- Model Trained --')

    # evaluation of performance on different sparisity levels
    print('\n-- Evaluating--')
    mse_per_sparsity = evaluate_for_all_spars(model, X_test, Y_test, spars_params, arch)
    mu = round(sum(mse_per_sparsity)/len(mse_per_sparsity))
    headings = ['Sparsity >'] + [str(s) for s in np.linspace(0, 0.9, 10, dtype=np.float16)] + ['Mean']
    print(
        'MSE Loss per Sparsity',
        tabulate([['mcVAE'] + mse_per_sparsity + [mu]], headers=headings, tablefmt='mixed_grid', numalign='right'),
        sep='\n'
    )

    print('\n-- Saving Model--')
    # save model
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = f'./logs/{args.name}-{now}'
    os.makedirs(path, exist_ok=True)
    fname = 'mcVAE_weights.pt'
    torch.save(model.state_dict(), os.path.join(path, fname))
    # save hyperparameters
    fname = 'hparams.json'
    hparams = {'arch': deepcopy(arch), 'params': deepcopy(params), 'spars_params': deepcopy(spars_params)}
    hparams['arch']['nonlinearity'] = arch['activation'].__class__.__name__
    hparams['spars_params']['sparsity_scheduler'] = spars_params['sparsity_scheduler'].__class__.__name__
    with open(os.path.join(path, fname), 'w') as f:
        json.dump(hparams, f)

    print('\n\nYour Model is trained 🥳')

if __name__ == '__name__':
    main()
