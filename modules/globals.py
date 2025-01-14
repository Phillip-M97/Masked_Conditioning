import torch
import torch.nn as nn
import pandas as pd
import os
if os.getcwd().split('/')[-1] != '03_mcVAE' and os.getcwd().split('\\')[-1] != '03_mcVAE':
    os.chdir('..')

# global variables
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = './data/car_data/'
MODEL_PATH = './models/'
REF_POINTS = 'extended/ref_points_with_dc.csv'
DRAG_COEFFS = 'extended/drag_coeffs_with_mtcb.csv'
FULL_DATA = 'full_dataset_enc.csv'
LOGFILE = 'logging.csv'

# vehicle information
data = pd.read_csv(DATA_PATH + REF_POINTS, usecols=[
    'manufacturer',
    'type',
    'class',
    'buzzwords1',
    'buzzwords2',
    'buzzwords3',
    'drag_coeff']).dropna()

info = {
    'manufacturer': list(data.manufacturer.unique()),
    'type': list(data.type.unique()),
    'class': list(data.loc[:,'class'].unique()),
    'buzzword': list(pd.concat([data.buzzwords1,
                                data.buzzwords2,
                                data.buzzwords3],
                                axis=0).unique()),
    'dc_min': data.drag_coeff.min(),
    'dc_max': data.drag_coeff.max()
}
del data

# model architecture
arch = {
    'rp_dim': 42,                           # ref points input dimension
    'dc_dim': 1,                            # drag coeff input dimension
    'm_dim': len(info['manufacturer']),     # manufacturer input dimension
    't_dim': len(info['type']),             # type input dimension
    'c_dim': len(info['class']),            # class input dimension
    'b_dim': len(info['buzzword']),         # buzzword input dimension
    'dc_enc_dim': 6,                        # positional encoding for more impact of the drag coeff
    'rp_embed_dim': 74,                     # embedding layer for reference points
    'cond_embed_dim': 11,                   # embedding layer for condition
    'hidden_dim': [100, 100, 100, 100],     # encoder/decoder layers
    'latent_dim': 11,                       # latent, mean, logvar dimension
    'activation': nn.SELU(),                # activation function
}

# parameters
params = {
    'batch_size': 64,                       # batch size
    'epochs': None,                         # number of epochs
    'lr': None,                             # learning rate for adam optimizer
    'betas': (.9, .999),                    # default betas for adam optimizer
    'kld_weight': None,                     # kld weight - similar to beta-VAE
    'weight_decay': None,                   # weight decay for adam optimizer
}

# parameters for sparsity schedulers
spars_params = {
    'sparsity_scheduler': None,             # scheduler type
    'sparsity': 0.,                         # initial sparsity
    'step_size': None,                      # step size for StepSparsity only
    'target_sparsity': 1.,                  # sparsity that shall be reached at sparsity_epoch
    'target_epoch': -1,                     # epoch where target_sparsity shall be reached
}

# condition template
cdict_template = {
    'manufacturer': None,
    'type': None,
    'class': None,
    'buzzword1': None,
    'buzzword2': None,
    'buzzword3': None,
    'drag_coeff': None,
}