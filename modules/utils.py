import copy
import csv
import random
import json

from modules.globals import *
from modules.mcvae import mcVAE, Mask, PosEnc
from modules.emb_mcvae import ConditionMasking
from modules.mlp import MLP
from modules.sparsity_scheduler import *
from modules.conv_mcvae import DiagonalGaussianDistribution
from modules.discriminator_loss import step_disc, adversarial_loss, NLayerDiscriminator

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from torchinfo import summary
from torch.distributions.multivariate_normal import MultivariateNormal
from matplotlib import colors, cm
from openTSNE import TSNE
from PIL import Image
import threading
import math
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance

from modules.data import RefPointData, RefPointDataCarEmb, BikeSketchDataset

def conditions_to_cdict(*args) -> dict:
    ''' Converts conditions to a valid cdict or creates random cdict if no args are given.
    '''
    assert len(args) <= 7, 'More than 7 conditions are not valid'

    dict = cdict_template.copy()

    # random condition if no cdict is given
    if len(args) == 0:
        dict.update({
                'manufacturer': random.choice(info['manufacturer']),
                'type': random.choice(info['type']),
                'class': random.choice(info['class']),
        })
        while(dict['buzzword1'] == dict['buzzword2'] or dict['buzzword2'] == dict['buzzword3'] or dict['buzzword1'] == dict['buzzword3']):
            dict.update({
                'buzzword1': random.choice(info['buzzword']),
                'buzzword2': random.choice(info['buzzword']),
                'buzzword3': random.choice(info['buzzword']),
                'drag_coeff': random.uniform(info['dc_min'], info['dc_max'])
            })

    # manufacturer, type, class
    for key in list(dict.keys())[:3]:
        for arg in args:
            if arg in info[key]:
                assert dict[key] is None, 'No multiple values possible for ' + str(key)
                dict[key] = arg
        
    # buzzwords
    for arg in args:
        if arg in info['buzzword']:
            assert (dict['buzzword1'] is None or dict['buzzword2'] is None or dict['buzzword3'] is None), 'Max 3 values possible for ' + 'buzzword'
            assert (dict['buzzword1'] != arg and dict['buzzword2'] != arg), 'No equal values allowed for ' + 'buzzword'

            if dict['buzzword1'] is None:
                dict['buzzword1'] = arg
            elif dict['buzzword2'] is None:
                dict['buzzword2'] = arg
            elif dict['buzzword3'] is None:
                dict['buzzword3'] = arg

    # drag coeff
    for arg in args:
        if type(arg) == float:
            assert dict['drag_coeff'] is None, 'No multiple values possible for ' + 'Drag Coeff'
            dict['drag_coeff'] = arg

    return dict

def df_to_cdict(df: pd.DataFrame) -> dict:
    ''' Converts single row DataFrame to a valid cdict.
    '''
    assert df.shape[0] == 1, 'DataFrame with only a single row is allowed'

    cdict = {
            'manufacturer': None,
            'type': None,
            'class': None,
            'buzzword1': None,
            'buzzword2': None,
            'buzzword3': None,
            'drag_coeff': None
            }
    
    cdict['manufacturer'] = df.loc[:,'manufacturer'].values[0]
    cdict['type'] = df.loc[:,'type'].values[0]
    cdict['class'] = df.loc[:,'class'].values[0]
    cdict['buzzword1'] = df.loc[:,'buzzwords1'].values[0]
    cdict['buzzword2'] = df.loc[:,'buzzwords2'].values[0]
    cdict['buzzword3'] = df.loc[:,'buzzwords3'].values[0]
    cdict['drag_coeff'] = df.loc[:,'drag_coeff'].values[0]

    return cdict

def cdict_to_df(cdict: dict) -> pd.DataFrame:
    ''' Converts valid cdict to a single row DataFrame.
    '''
    return pd.DataFrame({key: [val] for key, val in cdict.items()})

def enc(df: pd.DataFrame) -> np.ndarray:
    ''' One-Hot-Encodes DataFrame to ndarray.
    '''

    arr_enc = np.zeros((df.shape[0], arch['m_dim'] + arch['t_dim'] + arch['c_dim'] + arch['b_dim'] + arch['dc_dim']))

    for i in range(df.shape[0]):
        row = df.iloc[i,:]

        m_idx = info['manufacturer'].index(row.loc['manufacturer']) if row.loc['manufacturer'] is not None else None
        t_idx = info['type'].index(row.loc['type']) if row.loc['type'] is not None else None
        c_idx = info['class'].index(row.loc['class']) if row.loc['class'] is not None else None
        b1_idx = info['buzzword'].index(row.loc['buzzword1']) if row.loc['buzzword1'] is not None else None
        b2_idx = info['buzzword'].index(row.loc['buzzword2']) if row.loc['buzzword2'] is not None else None
        b3_idx = info['buzzword'].index(row.loc['buzzword3']) if row.loc['buzzword3'] is not None else None

        m_arr = np.eye(arch['m_dim'], dtype=float)[[m_idx]] if m_idx is not None else np.zeros((1, arch['m_dim']), dtype=float)
        t_arr = np.eye(arch['t_dim'], dtype=float)[[t_idx]] if t_idx is not None else np.zeros((1, arch['t_dim']), dtype=float)
        c_arr = np.eye(arch['c_dim'], dtype=float)[[c_idx]] if c_idx is not None else np.zeros((1, arch['c_dim']), dtype=float)
        b1_arr = np.eye(arch['b_dim'], dtype=bool)[[b1_idx]] if b1_idx is not None else np.zeros((1, arch['b_dim']), dtype=bool)
        b2_arr = np.eye(arch['b_dim'], dtype=bool)[[b2_idx]] if b2_idx is not None else np.zeros((1, arch['b_dim']), dtype=bool)
        b3_arr = np.eye(arch['b_dim'], dtype=bool)[[b3_idx]] if b3_idx is not None else np.zeros((1, arch['b_dim']), dtype=bool)
        b_arr = np.logical_or(np.logical_or(b1_arr, b2_arr), b3_arr)
        dc_arr = np.array(row.loc['drag_coeff'], dtype=float).reshape(1, -1) if row.loc['drag_coeff'] is not None else np.zeros((1, 1), dtype=float)

        arr_enc[i] = np.concatenate([m_arr, t_arr, c_arr, b_arr, dc_arr], axis=1, dtype=float)
    
    return arr_enc

def dec(arr: np.ndarray) -> pd.DataFrame:
    ''' Decodes One-Hot-Encoded ndarray to DataFrame.
    '''
    
    df_dec = pd.DataFrame(columns=cdict_template.keys())
    
    for i in range(arr.shape[0]):

        m_arr, t_arr, c_arr, b_arr, dc_arr = np.hsplit(arr[i], [
            arch['m_dim'],
            arch['m_dim'] + arch['t_dim'],
            arch['m_dim'] + arch['t_dim'] + arch['c_dim'],
            arch['m_dim'] + arch['t_dim'] + arch['c_dim'] + arch['b_dim'],
        ])

        m_idx = m_arr.argmax() if m_arr.any() else None
        t_idx = t_arr.argmax() if t_arr.any() else None
        c_idx = c_arr.argmax() if c_arr.any() else None

        b1_idx = np.nonzero(b_arr)[0][0] if len(list(np.nonzero(b_arr)[0])) > 0 else None
        b2_idx = np.nonzero(b_arr)[0][1] if len(list(np.nonzero(b_arr)[0])) > 1 else None
        b3_idx = np.nonzero(b_arr)[0][2] if len(list(np.nonzero(b_arr)[0])) > 2 else None

        m = info['manufacturer'][m_idx] if m_idx is not None else None
        t = info['type'][t_idx] if t_idx is not None else None
        c = info['class'][c_idx] if c_idx is not None else None
        b1 = info['buzzword'][b1_idx] if b1_idx is not None else None
        b2 = info['buzzword'][b2_idx] if b2_idx is not None else None
        b3 = info['buzzword'][b3_idx] if b3_idx is not None else None
        dc = dc_arr.item() if dc_arr.item() != 0 else None

        new_row_cdict = conditions_to_cdict(m, t, c, b1, b2, b3, dc)
        new_row_df = cdict_to_df(new_row_cdict)
    
        df_dec = pd.concat([df_dec, new_row_df], axis=0, ignore_index=True)

    return df_dec

def vae_loss(x, x_hat, mean, logvar, kld_weight):
    ''' Sum of reproduction loss (mse) and kullback-leibler-divergence (kld).
        Kld weight is similar to the beta of a beta-VAE.
    '''
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    kld = torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1))
    return (1 - kld_weight) * reproduction_loss, kld * kld_weight

def vae_loss_mean(x, x_hat, mean, logvar, kld_weight):
    ''' Sum of reproduction loss (mse) and kullback-leibler-divergence (kld).
        Kld weight is similar to the beta of a beta-VAE.
        Applies mean instead of sum reduction
    '''
    reproduction_loss = ((x.contiguous() - x_hat.contiguous())**2).mean()
    kld = 0.5 * torch.sum(torch.pow(mean, 2) + logvar.exp() - 1.0 - logvar, dim=[1, 2, 3])
    kld = torch.sum(kld)/kld.shape[0]
    return (1 - kld_weight) * reproduction_loss, kld * kld_weight

def plot_loss(train_hist: np.ndarray, val_hist: np.ndarray, lr_hist: np.ndarray, spars_hist: np.ndarray, skip: int, sup_title: str=None, is_adv: bool=False) -> None:
    skip = skip if len(train_hist) > skip else 0
    x = list(range(len(train_hist)))[skip:]

    plt_train, plt_val = train_hist[skip:,0], val_hist[skip:,0]
    plt_train_mse, plt_val_mse = train_hist[skip:,1], val_hist[skip:,1]
    plt_train_kld, plt_val_kld = train_hist[skip:,2], val_hist[skip:,2]

    if is_adv:
        plt_train_adv, plt_val_adv = train_hist[skip:, 3], val_hist[skip:, 3]

    fig, axs = plt.subplots(nrows=2, ncols=2, sharey='none', figsize=(12, 8))
    if sup_title:
        fig.suptitle(sup_title)

    axs[0][0].set_title('Total Loss')
    axs[0][0].plot(x, plt_train, label='Train')
    axs[0][0].plot(x, plt_val, label='Test')
    axs[0][0].legend(loc='upper right')
    axs[0][0].set_ylabel('Loss')

    axs[0][1].set_title('MSE')
    axs[0][1].plot(x, plt_train_mse, label='Train')
    axs[0][1].plot(x, plt_val_mse, label='Test')
    axs[0][1].legend(loc='upper right')

    axs[1][0].set_title('KLD')
    axs[1][0].plot(x, plt_train_kld, label='Train')
    axs[1][0].plot(x, plt_val_kld, label='Test')
    axs[1][0].legend(loc='upper right')
    axs[1][0].set_xlabel('Epoch')
    axs[1][0].set_ylabel('Loss')

    if not is_adv:
        axs[1][1].set_title('Scheduler')
        spars = axs[1][1].plot(list(range(len(spars_hist))), spars_hist, c='tab:green', label='Sparsity')
        axs[1][1].set_ylim(0, 1)
        axs[1][1].set_xlim(0, len(spars_hist))
        axs[1][1].set_xlabel('Epoch')
        axs[1][1].set_ylabel('Sparsity')
        

        if lr_hist is not None:
            lr_axs = axs[1][1].twinx()
            lr = lr_axs.plot(list(range(len(lr_hist))), lr_hist, c='tab:gray', label='Learning Rate')
            lr_axs.set_ylabel('Learning Rate')
            axs[1][1].legend(lr+spars, [lr[0].get_label(), spars[0].get_label()], loc='best')
        else:
            axs[1][1].legend(loc='best')
    else:
        axs[1][1].set_title('Adversarial')
        axs[1][1].plot(x, plt_train_adv, label='Train')
        axs[1][1].plot(x, plt_val_adv, label='Test')
        axs[1][1].legend(loc='upper right')
        axs[1][1].set_xlabel('Epoch')
        axs[1][1].set_ylabel('Loss')


    plt.show()

def train_step(train_loader: torch.utils.data.dataloader.DataLoader, model: mcVAE, loss_fn, optimizer, kld_weight, mean_reduce: bool=False) -> list:
    model.train()
    train_loss, train_mse_loss, train_kld_loss = 0, 0, 0

    for x, y in train_loader:
        optimizer.zero_grad()

        mean, logvar, _, _, x_hat = model(x, y)
        mse_loss, kld_loss = loss_fn(x, x_hat, mean, logvar, kld_weight)
        loss = mse_loss + kld_loss

        loss.backward()
        optimizer.step()

        if torch.isnan(loss):
            loss = torch.tensor(100000, dtype=torch.float, device=DEVICE)
        if torch.isnan(mse_loss):
            mse_loss = torch.tensor(100000, dtype=torch.float, device=DEVICE)
        if torch.isnan(kld_loss):
            kld_loss = torch.tensor(100000, dtype=torch.float, device=DEVICE)
        
        if mean_reduce:
            train_loss += loss.detach().cpu().item() * len(x)
            train_mse_loss += mse_loss.detach().cpu().item() * len(x)
            train_kld_loss += kld_loss.detach().cpu().item() * len(x)
        else:
            train_loss += loss.detach().cpu().item() * len(x)
            train_mse_loss += mse_loss.detach().cpu().item() * len(x)
            train_kld_loss += kld_loss.detach().cpu().item() * len(x)

    train_loss /= len(train_loader.dataset)
    train_mse_loss /= len(train_loader.dataset)
    train_kld_loss /= len(train_loader.dataset)

    return [train_loss, train_mse_loss, train_kld_loss]

def train_step_adversarial_vae(train_loader: DataLoader, model: mcVAE, vae_loss_fn, optimizer_vae: torch.optim.Optimizer, optimizer_disc: torch.optim.Optimizer, discriminator: nn.Module, kld_weight: float, adv_start: int, adv_weight: float, global_step: int) -> list:
    model.train()
    train_loss, train_mse_loss, train_kld_loss, train_adv_loss = 0, 0, 0, 0

    for x, y in train_loader:
        global_step += 1
        optimizer_vae.zero_grad()

        mean, logvar, _, _, x_hat = model(x, y)
        mse_loss, kld_loss = vae_loss_fn(x, x_hat, mean, logvar, kld_weight)
        vae_loss = mse_loss + kld_loss
        adv_loss = adversarial_loss(x_hat, discriminator, vae_loss, global_step, model.out.weight, disc_start=adv_start, d_weight=adv_weight)

        loss = vae_loss + adv_loss
        loss.backward()
        optimizer_vae.step()

        _ = step_disc(x, x_hat, discriminator, optimizer_disc, global_step, disc_start=adv_start)

        train_loss += loss.detach().cpu().item() * len(x)
        train_mse_loss += mse_loss.detach().cpu().item() * len(x)
        train_kld_loss += kld_loss.detach().cpu().item() * len(x)
        train_adv_loss += adv_loss.detach().cpu().item() * len(x)
    
    train_loss /= len(train_loader.dataset)
    train_mse_loss /= len(train_loader.dataset)
    train_kld_loss /= len(train_loader.dataset)
    train_adv_loss /= len(train_loader.dataset)

    return [train_loss, train_mse_loss, train_kld_loss, train_adv_loss], global_step


def val_step(val_loader: torch.utils.data.dataloader.DataLoader, model: mcVAE, loss_fn, kld_weight, mean_reduce: bool=False) -> list:
    model.eval()
    val_loss, val_mse_loss, val_kld_loss = 0, 0, 0

    for x, y in val_loader:

        mean, logvar, _, _, x_hat = model(x, y)
        mse_loss, kld_loss = loss_fn(x, x_hat, mean, logvar, kld_weight)
        loss = mse_loss + kld_loss

        if torch.isnan(loss):
            loss = torch.tensor(100000, dtype=torch.float, device=DEVICE)
        if torch.isnan(mse_loss):
            mse_loss = torch.tensor(100000, dtype=torch.float, device=DEVICE)
        if torch.isnan(kld_loss):
            kld_loss = torch.tensor(100000, dtype=torch.float, device=DEVICE)

        if mean_reduce:
            val_loss += loss.detach().cpu().item() * len(x)
            val_mse_loss += mse_loss.detach().cpu().item() * len(x)
            val_kld_loss += kld_loss.detach().cpu().item() * len(x)
        else:
            val_loss += loss.detach().cpu().item() * len(x)
            val_mse_loss += mse_loss.detach().cpu().item() * len(x)
            val_kld_loss += kld_loss.detach().cpu().item() * len(x)

    val_loss /= len(val_loader.dataset)
    val_mse_loss /= len(val_loader.dataset)
    val_kld_loss /= len(val_loader.dataset)
    
    return [val_loss, val_mse_loss, val_kld_loss]

def val_step_adversarial_vae(val_loader: DataLoader, model: mcVAE, loss_fn_vae, kld_weight, discriminator: nn.Module, global_step: int, adv_start: int, adv_weight: float):
    model.eval()
    val_loss, val_mse_loss, val_kld_loss, val_adv_loss = 0, 0, 0, 0

    for x, y in val_loader:
        mean, logvar, _, _, x_hat = model(x, y)
        mse_loss, kld_loss = loss_fn_vae(x, x_hat, mean, logvar, kld_weight)
        vae_loss = mse_loss + kld_loss
        adv_loss = adversarial_loss(x_hat, discriminator, vae_loss, global_step, model.out.weight, disc_start=adv_start, d_weight=adv_weight)
        loss = vae_loss + adv_loss
        
        val_loss += loss.detach().cpu().item()*len(x)
        val_mse_loss += mse_loss.detach().cpu()*len(x)
        val_kld_loss += kld_loss.detach().cpu()*len(x)
        val_adv_loss += kld_loss.detach().cpu()*len(x)
    
    val_loss /= len(val_loader.dataset)
    val_mse_loss /= len(val_loader.dataset)
    val_kld_loss /= len(val_loader.dataset)
    val_adv_loss /= len(val_loader.dataset)

    return [val_loss, val_mse_loss, val_kld_loss, val_adv_loss]
        

def train(model: mcVAE, train_loader: torch.utils.data.dataloader.DataLoader, val_loader: torch.utils.data.dataloader.DataLoader, verbose: int = 2, train_params: dict=None, train_spars_params: dict=None) -> tuple:
    if train_params is None:
        train_params = params
    if train_spars_params is None:
        train_spars_params = spars_params
    
    mean_reduce = train_params.get('mean_reduce', False)
    if not mean_reduce:
        loss_fn = vae_loss
    else:
        print('Using mean reduction')
        loss_fn = vae_loss_mean
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_params['lr'], weight_decay=train_params['weight_decay'], betas=train_params['betas'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    assert train_spars_params['sparsity_scheduler'] is not None, 'no sparsity scheduler defined'
    spars_scheduler = train_spars_params['sparsity_scheduler'](model, arch, train_params, train_spars_params)
    
    spars_hist = spars_scheduler.get_sparsity()
    lr_hist = [train_params['lr']]

    train_hist, val_hist = [], []
    for epoch in range(1, train_params['epochs']+1):
        
        train_loss = train_step(train_loader, model, loss_fn, optimizer, train_params['kld_weight'], mean_reduce)
        train_hist.append(train_loss)

        val_loss = val_step(val_loader, model, loss_fn, train_params['kld_weight'], mean_reduce) if val_loader is not None else [0, 0, 0]
        val_hist.append(val_loss)

        if verbose >= 2 and not (epoch) % 10:
            print(
                f'Epoch: {epoch:5}  ' +
                f'Train Loss: {train_loss[0]:5.3f}  ' +
                f'MSE: {train_loss[1]:5.3f}  ' +
                f'KLD: {train_loss[2]:5.3f}  ' +
                f'Val Loss: {val_loss[0]:5.3f}  ' +
                f'MSE: {val_loss[1]:5.3f}  ' +
                f'KLD: {val_loss[2]:5.3f}  ' +
                f'Lr: {lr_scheduler.get_last_lr()[0]:.5f}  ' +
                f'Spars: {spars_scheduler.get_last_sparsity():.5f}'
            )

        lr_scheduler.step()
        spars_scheduler.step()
        lr_hist.append(lr_scheduler.get_last_lr()[0])

    train_hist, val_hist = np.array(train_hist), np.array(val_hist)
    spars_hist, lr_hist = np.array(spars_hist), np.array(lr_hist)
    
    if verbose >= 0:
        print(
            '\nCurrent Loss\t>>> ' +
            f'Train Loss: {train_hist[-1, 0]:5.3f}  ' +
            f'MSE: {train_hist[-1, 1]:5.3f}  ' +
            f'KLD: {train_hist[-1, 2]:5.3f}  ' +
            f'Val Loss: {val_hist[-1, 0]:5.3f}  ' +
            f'MSE: {val_hist[-1, 1]:5.3f}  ' +
            f'KLD: {val_hist[-1, 2]:5.3f}'
            )
        
    if verbose >= 1:
        print(
            'Best Loss\t>>> ' +
            f'Train Loss: {train_hist[:,0].min():.3f}  ' +
            f'MSE: {train_hist[train_hist[:,0].argmin(),1]:.3f}  ' +
            f'KLD: {train_hist[train_hist[:,0].argmin(),2]:.3f}  ' +
            f'Val Loss: {val_hist[:,0].min():.3f}  ' +
            f'MSE: {val_hist[val_hist[:,0].argmin(),1]:.3f}  ' +
            f'KLD: {val_hist[val_hist[:,0].argmin(),2]:.3f}  ' +
            f'\n\t\t    at Epoch: {np.array(train_hist)[:,0].argmin()}' +
            f'\t\t\t       at Epoch: {np.array(val_hist)[:,0].argmin()}'
        )

    return train_hist, val_hist, lr_hist, spars_hist

def train_adversarial(model: mcVAE, train_loader: DataLoader, val_loader: DataLoader, train_params: dict, train_spars_params: dict, img_channels: int, start_channels: int, verbose: int=2) -> tuple:
    loss_fn = vae_loss_mean

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_params['lr'], weight_decay=train_params['weight_decay'], betas=train_params['betas'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    discriminator = NLayerDiscriminator(img_channels=img_channels, n_layers=len(model.enc), start_channels=start_channels).to(DEVICE)
    disc_optim = torch.optim.AdamW(discriminator.parameters(), lr=train_params['lr'])

    spars_scheduler = train_spars_params['sparsity_scheduler'](model, arch, train_params, train_spars_params)
    spars_hist = spars_scheduler.get_sparsity()
    lr_hist = [train_params['lr']]
    train_hist, val_hist = [], []
    global_step = 0

    for epoch in range(1, train_params['epochs']+1):
        train_loss, global_step = train_step_adversarial_vae(train_loader, model, loss_fn, optimizer, disc_optim, discriminator, train_params['kld_weight'], train_params['adv_start'], train_params['adv_weight'], global_step)
        train_hist.append(train_loss)
        val_loss = val_step_adversarial_vae(val_loader, model, loss_fn, train_params['kld_weight'], discriminator, global_step, train_params['adv_start'], train_params['adv_weight'])
        val_hist.append(val_loss)

        lr_scheduler.step()
        spars_scheduler.step()
        lr_hist.append(lr_scheduler.get_last_lr()[0])

        if verbose >= 2 and not (epoch) % 10:
            print(
                f'Epoch: {epoch:5}  ' +
                f'Train Loss: {train_loss[0]:5.3f}  ' +
                f'MSE: {train_loss[1]:5.3f}  ' +
                f'KLD: {train_loss[2]:5.3f}  ' +
                f'ADV: {train_loss[3]:5.3f}  ' +
                f'Val Loss: {val_loss[0]:5.3f}  ' +
                f'MSE: {val_loss[1]:5.3f}  ' +
                f'KLD: {val_loss[2]:5.3f}  ' +
                f'ADV: {val_loss[3]:5.3f}  ' +
                f'Lr: {lr_scheduler.get_last_lr()[0]:.5f}  ' +
                f'Spars: {spars_scheduler.get_last_sparsity():.5f}'
            )

    train_hist, val_hist = np.array(train_hist), np.array(val_hist)
    spars_hist, lr_hist = np.array(spars_hist), np.array(lr_hist)

    if verbose >= 0:
        print(
                '\nCurrent Loss\t>>> ' +
                f'Train Loss: {train_hist[-1, 0]:5.3f}  ' +
                f'MSE: {train_hist[-1, 1]:5.3f}  ' +
                f'KLD: {train_hist[-1, 2]:5.3f}  ' +
                f'ADV: {train_hist[-1, 3]:5.3f}  ' +
                f'Val Loss: {val_hist[-1, 0]:5.3f}  ' +
                f'MSE: {val_hist[-1, 1]:5.3f}  ' +
                f'KLD: {val_hist[-1, 2]:5.3f}' +
                f'ADV: {val_hist[-1, 3]:5.3f}  '
            )
    
    if verbose >= 1:
        print(
            'Best Loss\t>>> ' +
            f'Train Loss: {train_hist[:,0].min():.3f}  ' +
            f'MSE: {train_hist[train_hist[:,0].argmin(),1]:.3f}  ' +
            f'KLD: {train_hist[train_hist[:,0].argmin(),2]:.3f}  ' +
            f'ADV: {train_hist[train_hist[:,0].argmin(),3]:.3f}  ' +
            f'Val Loss: {val_hist[:,0].min():.3f}  ' +
            f'MSE: {val_hist[val_hist[:,0].argmin(),1]:.3f}  ' +
            f'KLD: {val_hist[val_hist[:,0].argmin(),2]:.3f}  ' +
            f'ADV: {val_hist[val_hist[:,0].argmin(),3]:.3f}  ' +
            f'\n\t\t    at Epoch: {np.array(train_hist)[:,0].argmin()}' +
            f'\t\t\t       at Epoch: {np.array(val_hist)[:,0].argmin()}'
        )
    return train_hist, val_hist, lr_hist, spars_hist


def logging(train_hist: list, val_hist: list, model: mcVAE, train_time: float, logfile: str, train_params: dict=None, train_spars_params: dict=None, train_arch: dict=None) -> None:
    train_hist, val_hist = np.array(train_hist), np.array(val_hist)

    input_size = ((train_params['batch_size'], train_arch['rp_dim']),
                  (train_params['batch_size'], train_arch['m_dim'] + train_arch['t_dim'] + train_arch['c_dim'] + train_arch['b_dim'] + train_arch['dc_dim']))
    model_stats = summary(model, input_size=input_size)
    
    log = [
        round(val_hist[:,0].min(), 4),                              # min. val loss
        val_hist[:,0].argmin(),                                     # at epoch
        round(train_hist[:,0].min(), 4),                            # min. train loss
        train_hist[:,0].argmin(),                                   # at epoch
        model_stats.total_params,                                   # num model parameters
        model_stats.total_mult_adds,                                # num mult adds

        # architecture
        train_arch['rp_embed_dim'],
        train_arch['cond_embed_dim'],
        train_arch['hidden_dim'][0],
        train_arch['hidden_dim'][1] if len(train_arch['hidden_dim']) >= 2 else 0,
        train_arch['hidden_dim'][2] if len(train_arch['hidden_dim']) >= 3 else 0,
        train_arch['hidden_dim'][3] if len(train_arch['hidden_dim']) >= 4 else 0,
        train_arch['latent_dim'],
        train_arch['dc_enc_dim'],
        train_arch['activation'],
        train_spars_params['sparsity'],
        train_spars_params['sparsity_scheduler'],

        round(np.array(val_hist).argmin() * (train_time/60) / train_params['epochs'], 2),     # approximated training time for best val loss
        train_params['batch_size'],
        train_params['epochs'],
        train_params['lr'],
        train_params['betas'][0],
        train_params['betas'][1],
        train_params['kld_weight'],
        train_params['weight_decay'],
    ]

    with open('data/' + logfile, 'a', newline='') as logfile:
        csv.writer(logfile).writerow(log)

def plot_silhouette(coordinates: np.ndarray, title: str=None, show:bool=True) -> None:
    ''' Plots 2D Silhouette of a vehicle given as reference points coordinates.
        Green frame shows the min and max coordinates of the original dataset.
    '''
    # points
    x_coord, y_coord = coordinates[::2], coordinates[1::2]

    # lines
    x_coord_line1, y_coord_line1 = x_coord[:14], y_coord[:14]
    x_coord_line2, y_coord_line2 = x_coord[15:17], y_coord[15:17]

    # rear wheel
    o_rear_wheel = (x_coord[14], y_coord[14])

    fig, axs = plt.subplots(nrows=1, ncols=1, subplot_kw=dict(aspect='equal'))

    if title:
        fig.suptitle(title)

    axs.scatter(x_coord, y_coord, s=10, color='tab:grey')
    axs.plot(x_coord_line1, y_coord_line1, lw=1, color='tab:blue')
    axs.plot(x_coord_line2, y_coord_line2, lw=1, color='tab:blue')
    axs.add_patch(Circle((0,0), 370, fill=False, color='tab:blue'))
    axs.add_patch(Circle(o_rear_wheel, 370, fill=False, color='tab:blue'))

    # xticks, yticks
    axs.set_xticks(np.linspace(-1104, 5123, 10))
    axs.set_yticks(np.linspace(-370, 1880, 5))
    xticks = axs.get_xticks()
    yticks = axs.get_yticks()
    axs.set_xlim((xticks[0]-150, xticks[-1]+150))
    axs.set_ylim(yticks[0]-150, yticks[-1]+150)
    
    axs.set_xticklabels([])
    axs.set_yticklabels([])

    axs.grid()
    axs.axhline(y=yticks[0], color='seagreen', linestyle='-', linewidth=1)
    axs.axhline(y=yticks[-1], color='seagreen', linestyle='-', linewidth=1)
    axs.axvline(x=xticks[0], color='seagreen', linestyle='-', linewidth=1)
    axs.axvline(x=xticks[-1], color='seagreen', linestyle='-', linewidth=1)
    
    if show:
        plt.show()
    else:
        plt.close()

    return fig

def latent_space(model: mcVAE, X: np.ndarray, Y_enc: np.ndarray) -> torch.Tensor:
    ''' Returns latent space of a model and given data.
    '''
    
    X_t = torch.tensor(X, dtype=torch.float, device=DEVICE)
    Y_t = torch.tensor(Y_enc, dtype=torch.float, device=DEVICE)

    model.eval()
    with torch.no_grad():
        Z = model(X_t, Y_t)[3]

    return Z

multi_normal_lock = threading.Lock()
def sample(arch: dict, set_seed: bool=True) -> torch.Tensor:
    ''' Returns random sample from standard normal distribution.
    '''

    if set_seed:
        torch.manual_seed(42)

    with multi_normal_lock:
        mean = torch.zeros((1, arch['latent_dim']), dtype=torch.float, device=DEVICE)
        cov = torch.eye(arch['latent_dim'], dtype=torch.float, device=DEVICE)
        multi_normal = MultivariateNormal(loc=mean, covariance_matrix=cov)
    
    z = multi_normal.sample()

    return z

def sample_2d(arch: dict, width: int, height: int, set_seed: bool=True):
    if set_seed:
        torch.manual_seed(42)
    mean = torch.zeros((1, arch['latent_dim'], width, height))
    logvar = torch.ones_like(mean)
    dist = DiagonalGaussianDistribution(mean, logvar)
    z = dist.sample()
    return z
    
def plot_latent_space(model: mcVAE, Y_enc_df: pd.DataFrame, Z: torch.Tensor, z: torch.Tensor = None, cdict: dict = None) -> np.ndarray:
    ''' Plots latent space of model optionally with given sample.
    '''
    assert not (z is None) ^ (cdict is None), 'both arguments z and cdict or neither must be given'

    # if a condition is given
    cond = True if cdict else False
    
    # all latent vecotrs for embedding
    Z_ = torch.concat([Z, z], dim=0).detach().cpu() if cond else Z.detach().cpu()

    # 2d embedding
    tsne = TSNE(
            perplexity=30,
            metric="euclidean",
            n_jobs=8,
            random_state=42,
            verbose=False,
        )
    
    Z_embed = tsne.fit(Z_)
    Z_rp_embed = Z_embed[:len(Z)]

    if cond:
        sample_embed = Z_embed[-1]

        model.eval()
        with torch.no_grad():
            y = torch.tensor(enc(cdict_to_df(cdict)), dtype=torch.float, device=DEVICE)
            y_p_enc = model.pos_enc(y)
            y_emb = model.y_emb(y_p_enc)

            # silhouette
            x_hat = model.decode(z, y_emb).detach().cpu().squeeze(0).numpy() * 1000
            plot_silhouette(x_hat)

        # predict drag coeff
        dc_hat = predict_drag_coeff(x_hat)
    

    fig, axs = plt.subplots(figsize=(6, 6), nrows=1, ncols=1)

    # Drag Coeff colorbar
    cmap_dc = plt.cm.get_cmap('coolwarm')
    cax_dc = fig.add_axes([.95, .1, .02, .8])
    vmin = Y_enc_df.drag_coeff.min().round(1) - 0.1
    vmax = Y_enc_df.drag_coeff.max().round(1) + 0.1
    norm_dc = colors.Normalize(vmin=vmin, vmax=vmax)
    cbar_dc = fig.colorbar(cm.ScalarMappable(norm=norm_dc, cmap=cmap_dc), cax=cax_dc, orientation='vertical', label='Drag Coeff')
    ticks_dc = list(np.linspace(vmin, vmax, int((vmax-vmin+0.1)*10)).round(2))
    labels_dc = ticks_dc.copy()
    if cond:
        # add samples dc to colorbar
        ticks_dc.append(dc_hat)
        ticks_dc.sort()
        dc_idx = ticks_dc.index(dc_hat)
        labels_dc.insert(dc_idx, round(dc_hat, 2))
    cbar_dc.set_ticks(ticks_dc)
    cbar_dc.set_ticklabels(labels_dc)

    # latent space
    axs.set_title('Latent Space (' + str(arch['latent_dim']) + 'D)')
    axs.scatter(Z_rp_embed[:,0], Z_rp_embed[:,1], c=Y_enc_df.drag_coeff.values, cmap=cmap_dc, s=15, edgecolors='none', label='Ref Points Dataset')
    if cond:
        axs.scatter(sample_embed[0], sample_embed[1], c=cmap_dc(norm_dc(dc_hat)), s=200, marker='*', edgecolors='black', linewidth=.5, label='Sample')
    axs.legend(loc='upper left', bbox_to_anchor=(-0.45, 1.15))
    plt.show()

    return x_hat if cond else None

def show_sample(cdict, model, X, Y_enc, title: str=None):
    Z = latent_space(model, X, Y_enc)
    z = sample(arch)
    with torch.no_grad():
        y = torch.tensor(enc(cdict_to_df(cdict)), dtype=torch.float, device=DEVICE)
        y_p_enc = model.pos_enc(y)
        y_emb = model.y_emb(y_p_enc)
        x_hat = model.decode(z, y_emb).detach().cpu().squeeze(0).numpy() * 1000
        plot_silhouette(x_hat, title)

def predict_drag_coeff(x_hat: np.ndarray) -> float:
    ''' Predicts the drag coefficient of a vehicle given as reference points coordinates
        by using the MLP which was used to complete the dataset without drag coeffs.
    '''
    mlp = MLP()
    mlp.load_state_dict(torch.load(MODEL_PATH + 'drag_coeff_prediction'))

    x_hat = torch.tensor(x_hat, dtype=torch.float) / 1000

    if x_hat.dim() == 1:
        x_hat = x_hat.unsqueeze(0)

    mlp.eval()
    with torch.no_grad():
        dc_hat = mlp(x_hat).item()

    return dc_hat

def get_inference_test_sets(Y: torch.Tensor, c_arch: dict=None, c_spars_params: dict=None, mask_value: float=0, old: bool=True) -> torch.Tensor:
    ''' Creates testsets  (conditions only) for inference with different levels of sparsity.
        Given Y must be one-hot-enc.
    '''
    if c_arch is None:
        c_arch = arch
    if c_spars_params is None:
        c_spars_params = spars_params

    # check if Y has the right dimension
    if old:
        dims = [c_arch[d] for d in c_arch['cond_dims']]
        total_dim = sum(dims)
    else:
        total_dim = len(c_arch['cond_dims'])
    assert Y.shape[1] == total_dim, 'given Y doesn\'t have the right dimension'

    sparsities = [.0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
    if 'dc_dim' in c_arch.keys():
        pos_enc = PosEnc(c_arch['dc_dim'], c_arch['dc_enc_dim'])
    else:
        pos_enc = None

    if old:
        mask = Mask(c_arch, c_spars_params, mask_value=mask_value)
    else:
        mask = ConditionMasking(c_arch, c_spars_params)

    # positional encoding
    if pos_enc:
        Y_wo_dc = Y[:,:-1]
        dc = Y[:,-1].reshape(-1, 1)
        dc_enc = pos_enc(dc)
        Y_p_enc = torch.concat([Y_wo_dc, dc_enc], dim=1)
    else:
        Y_p_enc = Y

    # create a dataset for each sparsity
    dims = [c_arch[d] for d in c_arch['cond_dims']]
    if old:
        shape = (0, Y_p_enc.size(0), sum(dims))
    else:
        shape = (0, Y_p_enc.size(0), len(dims))
    Y_inf_testsets = torch.empty(size=shape, dtype=torch.float, device=DEVICE)

    for sparsity in sparsities:
        mask.p = sparsity
        Y_msk = mask(Y_p_enc)
        Y_inf_testsets = torch.concat([Y_inf_testsets, Y_msk.unsqueeze(0)], dim=0)

    return Y_inf_testsets

def evaluate_inference(model: mcVAE, X: np.ndarray, Y: np.ndarray, c_arch: dict=None, c_spars_params: dict=None, mask_value: float=0, old: bool=True) -> list:
    ''' Returns MSE between given dataset and output of model for given condition while inference for different sparsity levels.
    '''
    if c_arch is None:
        c_arch = arch

    # to tensor
    X = torch.tensor(X, dtype=torch.float, device=DEVICE)
    Y = torch.tensor(Y, dtype=torch.float, device=DEVICE)

    # latent samples
    Z = torch.concat([sample(c_arch) for _ in range(len(Y))], dim=0)
    
    # create test datasets for inference
    Y_per_sparsity = get_inference_test_sets(Y, c_arch, c_spars_params, mask_value=mask_value, old=old)

    model.eval()
    with torch.no_grad():
        mse_per_sparsity = []
        for Y in Y_per_sparsity:
            Y_emb = model.y_emb(Y)

            # output
            X_hat = model.decode(Z, Y_emb)
            mse = nn.functional.mse_loss(X_hat, X, reduction='sum') / len(Y)

            mse_per_sparsity.append(mse.detach().cpu().round(decimals=3).item())
    
    return mse_per_sparsity

def get_conditioning_vector_embedding_biked(cond: dict, full_df: pd.DataFrame):
    CONDITION_MAPPINGS = {
        'BikeStyle': {
            'ROAD': 0, 'DIRT_JUMP': 1, 'POLO': 2, 'BMX': 3, 'MTB': 4, 'TOURING': 5, 'TRACK': 6, 'CRUISER': 7, 'COMMUTER': 8, 'CITY': 9, 'CYCLOCROSS': 10, 'OTHER': 11, 'TRIALS': 12, 'CHILDRENS': 13, 'TIMETRIAL': 14, 'DIRT': 15, 'CARGO': 16, 'HYBRID': 17, 'GRAVEL': 18, 'FAT': 19
        },
        'FrameSize': {
            'M': 0, 'XL': 1, 'XS': 2, 'L': 3, 'S': 4
        },
        'RimStyleFront': {
            'spoked': 0, 'trispoke': 1, 'disc': 2
        },
        'RimStyleRear': {
            'spoked': 0, 'trispoke': 1, 'disc': 2
        }
    }
    NUMERICAL_CONDITIONS = ['TeethChain']
    BINARY_CONDITIONS = ['BottleSeatTube', 'BottleDownTube', 'ForkType']

    encoded_dict = dict()
    for condition_name in cond.keys():
        if condition_name in NUMERICAL_CONDITIONS:
            encoded_dict[condition_name] = (cond[condition_name] - full_df[condition_name].min())/(full_df[condition_name].max() - full_df[condition_name].min())
        elif condition_name in BINARY_CONDITIONS:
            encoded_dict[condition_name] = cond[condition_name]
        else:
            encoded_dict[condition_name] = CONDITION_MAPPINGS[condition_name][cond[condition_name]]
    
    # sort dict
    order = ['BikeStyle', 'TeethChain', 'BottleSeatTube', 'BottleDownTube', 'FrameSize', 'RimStyleFront', 'RimStyleRear', 'ForkType']
    encoded_dict = {k: encoded_dict[k] for k in order}

    conditioning_vector = np.array(list(encoded_dict.values()))
    return conditioning_vector.astype(float)


def preprocess_biked_data(parameter_csv_path: str, normalize: bool=True, combine: bool=False, return_categorical: bool=False) -> pd.DataFrame:
    df_biked = pd.read_csv(parameter_csv_path)
    df_biked.drop(columns=[df_biked.columns[0], 'Bike index'], inplace=True)

    PREDICT_COLUMNS = ['x_rear_wheel_center', 'y_rear_wheel_center', 'x_BB', 'y_BB',
        'x_front_wheel_center', 'y_front_wheel_center', 'x_head_tube_top',
        'y_head_tube_top', 'x_rear_tube_connect_seat_tube',
        'y_rear_tube_connect_seat_tube', 'x_top_tube_connect_seat_tube',
        'y_top_tube_connect_seat_tube', 'x_top_tube_connect_head_tube',
        'y_top_tube_connect_head_tube', 'x_down_tube_connect_head_tube',
        'y_down_tube_connect_head_tube', 'x_stem_top', 'y_stem_top',
        'x_front_fork', 'y_front_fork', 'x_saddle_top', 'y_saddle_top',
        'x_seat_tube_top', 'y_seat_tube_top', 'Wheel diameter rear', 'Wheel diameter front']
    CONDITION_COLUMNS = [x for x in df_biked.columns if x not in PREDICT_COLUMNS]
    NUMERICAL_CONDITIONS = ['TeethChain']
    BINARY_CONDITIONS = ['BottleSeatTube', 'BottleDownTube', 'ForkType']
    CONDITION_MAPPINGS = {
        'BikeStyle': {
            'ROAD': 0, 'DIRT_JUMP': 1, 'POLO': 2, 'BMX': 3, 'MTB': 4, 'TOURING': 5, 'TRACK': 6, 'CRUISER': 7, 'COMMUTER': 8, 'CITY': 9, 'CYCLOCROSS': 10, 'OTHER': 11, 'TRIALS': 12, 'CHILDRENS': 13, 'TIMETRIAL': 14, 'DIRT': 15, 'CARGO': 16, 'HYBRID': 17, 'GRAVEL': 18, 'FAT': 19
        },
        'FrameSize': {
            'M': 0, 'XL': 1, 'XS': 2, 'L': 3, 'S': 4
        },
        'RimStyleFront': {
            'spoked': 0, 'trispoke': 1, 'disc': 2
        },
        'RimStyleRear': {
            'spoked': 0, 'trispoke': 1, 'disc': 2
        }
    }

    df_biked_points = df_biked[PREDICT_COLUMNS]
    df_biked_conditions = df_biked[CONDITION_COLUMNS]

    df_biked_conditions.columns = ['BikeStyle', 'TeethChain', 'BottleSeatTube', 'BottleDownTube', 'FrameSize', 'RimStyleFront', 'RimStyleRear', 'ForkType']
    df_biked_conditions_encoded = pd.get_dummies(df_biked_conditions, columns=['BikeStyle', 'FrameSize', 'RimStyleFront', 'RimStyleRear', 'ForkType'], dtype=int)
    posenc = PosEnc(1, 6)
    posenc = PosEnc(1, 6)
    teeth_t = torch.tensor(df_biked_conditions_encoded['TeethChain'].values).to(DEVICE)
    teeth_t = teeth_t.view((teeth_t.size(0), 1))
    teeth_encoded_t = posenc(teeth_t)
    teeth_encoded = teeth_encoded_t.detach().cpu().numpy()

    for i in range(teeth_encoded.shape[1]):
        df_biked_conditions_encoded[f'TeethChain_{i}'] = teeth_encoded[:, i]
    df_biked_conditions_encoded.drop(columns='TeethChain', inplace=True)

    df_conditions_tc_norm = df_biked_conditions.copy()
    for numerical_column in NUMERICAL_CONDITIONS:
        df_conditions_tc_norm[numerical_column] = (df_conditions_tc_norm[numerical_column] - df_conditions_tc_norm[numerical_column].min()) / (df_conditions_tc_norm[numerical_column].max() - df_conditions_tc_norm[numerical_column].min())

    for column_name in df_conditions_tc_norm.columns:
        if column_name in NUMERICAL_CONDITIONS + BINARY_CONDITIONS:
            continue
        df_conditions_tc_norm[column_name] = df_conditions_tc_norm[column_name].replace(CONDITION_MAPPINGS[column_name])
        df_conditions_tc_norm[column_name] = df_conditions_tc_norm[column_name].astype(int)

    normalization = {}
    if normalize:
        df_biked_points_normalized = (df_biked_points - df_biked_points.min()) / (df_biked_points.max() - df_biked_points.min())
        normalization = {x: {'max': df_biked_points[x].max(), 'min': df_biked_points[x].min()} for x in df_biked_points.columns}
    else:
        df_biked_points_normalized = df_biked_points
        normalization = {x: {'max': 1, 'min': 0} for x in df_biked_points.columns}
    
    if combine:
        if return_categorical:
            return pd.concat([df_biked_points_normalized, df_biked_conditions_encoded], axis=1), df_conditions_tc_norm, df_biked_conditions, normalization
        return pd.concat([df_biked_points_normalized, df_biked_conditions_encoded], axis=1), normalization
    else:
        if return_categorical:
            return df_biked_points_normalized, df_biked_conditions_encoded, df_conditions_tc_norm, df_biked_conditions, normalization
        return df_biked_points_normalized, df_biked_conditions_encoded, normalization
    
def plot_biked_sample(sample: np.array, normalization: dict, rear_center_idx: int=0, front_center_idx: int=2, x_max: int=1800, y_max: int=1200, show: bool=True):
    if len(sample.shape) == 2:
        sample = sample[0]
    
    unnormalized = []
    for i, s in enumerate(sample):
        c_min = normalization[list(normalization.keys())[i]]['min']
        c_max = normalization[list(normalization.keys())[i]]['max']
        c = s * (c_max - c_min) + c_min
        unnormalized.append(c)

    points = unnormalized[:-2]
    points = np.array([points[::2], points[1::2]])
    rear_wheel_diameter = unnormalized[-2]
    front_wheel_diameter = unnormalized[-1]

    fig, ax = plt.subplots()
    ax.scatter(points[0, :], points[1, :])
    
    front_wheel = plt.Circle((points[0, front_center_idx], points[1, front_center_idx]), front_wheel_diameter/2, fill=False, color='blue')
    rear_wheel = plt.Circle((points[0, rear_center_idx], points[1, rear_center_idx]), rear_wheel_diameter/2, fill=False, color='blue')
    ax.add_patch(front_wheel)
    ax.add_patch(rear_wheel)
    ax.set_aspect('equal')

    ax.set_ylim(0, y_max)
    ax.set_xlim(0, x_max)

    ax.set_axis_off()
    if show:
        plt.show()
    else:
        plt.close()
    return fig

def get_conditioning_vector_from_dict_biked(conditioning: dict, missing_value: int=0):
    posenc = PosEnc(1, 6)
    teeth_t = torch.tensor([conditioning.get('TeethChain', 0)]).to(DEVICE)
    teeth_t = teeth_t.view((teeth_t.size(0), 1))
    teeth_encoded_t = posenc(teeth_t)
    teeth_encoded = teeth_encoded_t.detach().squeeze().cpu().numpy()
    
    encoded_dict = {
        'BottleSeatTube': 1 if conditioning.get('BottleSeatTube', False) else 0,
        'BottleDownTube': 1 if conditioning.get('BottleDownTube', False) else 0,
        'BikeStyle_BMX': 1 if conditioning.get('BikeStyle', None) == 'BMX' else 0,
        'BikeStyle_CARGO': 1 if conditioning.get('BikeStyle', None) == 'CARGO' else 0,
        'BikeStyle_CHILDRENS': 1 if conditioning.get('BikeStyle', None) == 'CHILDRENS' else 0,
        'BikeStyle_CITY': 1 if conditioning.get('BikeStyle', None) == 'CITY' else 0,
        'BikeStyle_COMMUTER': 1 if conditioning.get('BikeStyle', None) == 'COMMUTER' else 0,
        'BikeStyle_CRUISER': 1 if conditioning.get('BikeStyle', None) == 'CRUISER' else 0,
        'BikeStyle_CYCLOCROSS': 1 if conditioning.get('BikeStyle', None) == 'CYCLOCROSS' else 0,
        'BikeStyle_DIRT': 1 if conditioning.get('BikeStyle', None) == 'DIRT' else 0,
        'BikeStyle_DIRT_JUMP': 1 if conditioning.get('BikeStyle', None) == 'DIRT_JUMP' else 0,
        'BikeStyle_FAT': 1 if conditioning.get('BikeStyle', None) == 'FAT' else 0,
        'BikeStyle_GRAVEL': 1 if conditioning.get('BikeStyle', None) == 'GRAVEL' else 0,
        'BikeStyle_HYBRID': 1 if conditioning.get('BikeStyle', None) == 'HYBRID' else 0,
        'BikeStyle_MTB': 1 if conditioning.get('BikeStyle', None) == 'MTB' else 0,
        'BikeStyle_OTHER': 1 if conditioning.get('BikeStyle', None) == 'OTHER' else 0,
        'BikeStyle_POLO': 1 if conditioning.get('BikeStyle', None) == 'POLO' else 0,
        'BikeStyle_ROAD': 1 if conditioning.get('BikeStyle', None) == 'ROAD' else 0,
        'BikeStyle_TIMETRIAL': 1 if conditioning.get('BikeStyle', None) == 'TIMETRAIL' else 0,
        'BikeStyle_TOURING': 1 if conditioning.get('BikeStyle', None) == 'TOURING' else 0,
        'BikeStyle_TRACK': 1 if conditioning.get('BikeStyle', None) == 'TRACK' else 0,
        'BikeStyle_TRIALS': 1 if conditioning.get('BikeStyle', None) == 'TRIALS' else 0,
        'FrameSize_L': 1 if conditioning.get('FrameSize', None) == 'L' else 0,
        'FrameSize_M': 1 if conditioning.get('FrameSize', None) == 'M' else 0,
        'FrameSize_S': 1 if conditioning.get('FrameSize', None) == 'S' else 0,
        'FrameSize_XL': 1 if conditioning.get('FrameSize', None) == 'XL' else 0,
        'FrameSize_XS': 1 if conditioning.get('FrameSize', None) == 'XS' else 0,
        'RimStyleFront_disc': 1 if conditioning.get('RimStyleFront', None) == 'disc' else 0,
        'RimStyleFront_spoked': 1 if conditioning.get('RimStyleFront', None) == 'spoked' else 0,
        'RimStyleFront_trispoke': 1 if conditioning.get('RimStyleFront', None) == 'trispoke' else 0,
        'RimStyleRear_disc': 1 if conditioning.get('RimStyleRear', None) == 'disc' else 0,
        'RimStyleRear_spoked': 1 if conditioning.get('RimStyleRear', None) == 'spoked' else 0,
        'RimStyleRear_trispoke': 1 if conditioning.get('RimStyleRear', None) == 'trispoke' else 0,
        'ForkType_0': 1 if conditioning.get('ForkType', -1) == 0 else 0,
        'ForkType_1': 1 if conditioning.get('ForkType', -1) == 1 else 0,
        'ForkType_2': 1 if conditioning.get('ForkType', -1) == 2 else 0,
        'TeethChain_0': teeth_encoded[0],
        'TeethChain_1': teeth_encoded[1],
        'TeethChain_2': teeth_encoded[2],
        'TeethChain_3': teeth_encoded[3],
        'TeethChain_4': teeth_encoded[4],
        'TeethChain_5': teeth_encoded[5],
    }
    conditioning_vector = np.array(list(encoded_dict.values()))
    return conditioning_vector.astype(float)

def get_Y_per_spars(Y: np.array, spars_params: dict, arch: dict, old: bool=True, min: float=0.0, max: float=0.9, step: float=0.1):
    c_spars_params = spars_params.copy()
    c_spars_params['sparsity'] = 0.
    c_spars_params['target_sparsity'] = 1.

    if old:
        arch['bst_dim'] = 1
        arch['bdt_dim'] = 1
        msk = Mask(arch, c_spars_params, c_spars_params['mask_value'])
    if not old:
        arch['bst_dim'] = 2
        arch['bdt_dim'] = 2
        msk = ConditionMasking(arch, c_spars_params)
    
    print('Set up done')

    y_per_spars = []
    for c_spars in np.linspace(min, max, int((max-min)/step)+1):
        msk.p = c_spars
        y_per_spars.append(msk(Y.clone()))
    return y_per_spars

def evaluate_for_all_spars(model: nn.Module, X: np.array, Y: np.array, spars_params: dict, arch: dict, old: bool=True, min: float=0.0, max: float=0.9, step: float=0.1):
    y_per_spars = get_Y_per_spars(Y, spars_params, arch, old, min, max, step)
    print(y_per_spars[0].size())
    x = torch.tensor(X, dtype=torch.float).to(DEVICE)
    z_sample = torch.concat([sample(arch) for _ in range(len(Y))], dim=0).to(DEVICE)

    model = model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        mse_per_sparsity = []
        for i, c_y in enumerate(y_per_spars):
            c_y_emb = model.y_emb(c_y.clone().to(DEVICE))
            X_hat = model.decode(z_sample, c_y_emb.clone())
            c_mse = nn.functional.mse_loss(X_hat, x.clone(), reduction='sum')/len(c_y)
            mse_per_sparsity.append(c_mse.detach().cpu().item())
    return mse_per_sparsity


def img_evaluate_for_all_spars(model: nn.Module, test_dl: DataLoader, spars_params: dict, arch: dict, min: float=0.0, max: float=0.9, step: float=0.1):
    model = model.to(DEVICE)
    model.eval()
    mse_per_sparsity = []
    msk = ConditionMasking(arch, spars_params)
    for c_spars in np.linspace(min, max, int((max-min)/step)+1):
        msk.p = c_spars
        c_mses = []
        for x, y in test_dl:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            latent_width = int(x.size(-2)/(math.pow(2, arch['num_levels'])))
            latent_height = int(x.size(-1)/(math.pow(2, arch['num_levels'])))
            z_sample = torch.concat([sample_2d(arch, latent_width, latent_height, set_seed=True) for _ in range(y.size(0))], dim=0).to(DEVICE)
            with torch.no_grad():
                c_y = msk(y)
                c_y = model.embed_y(z_sample, c_y)
                X_hat = model.decode(z_sample, c_y)
            c_mse = nn.functional.mse_loss(X_hat, x, reduction='mean').detach().cpu().item()
            c_mses.append(c_mse)
        mse_per_sparsity.append(sum(c_mses)/len(c_mses))
    return mse_per_sparsity

def diffusion_evaluate_for_all_spars(model: nn.Module, test_dl: DataLoader, spars_params: dict, arch: dict, min: float=0.0, max: float=0.9, step: float=0.1, num_batches: int=10) -> list:
    model = model.to(DEVICE)
    model.eval()
    mse_per_sparsity = []
    prev_training = model.msk.training
    prev_p = model.msk.p
    # msk = ConditionMasking(arch, spars_params)
    for c_spars in tqdm(np.linspace(min, max, int((max-min)/step)+1)):
        # msk.p = c_spars
        model.msk.training = True
        model.msk.p = c_spars
        c_mses = []
        for i, (x, y) in enumerate(test_dl):
            if i == num_batches:
                break
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            with torch.no_grad():
                X_hat = model.sample((x.size(2)*8, x.size(3)*8), y)
            x_decoded = model.decode(x)
            c_mse = nn.functional.mse_loss(X_hat, x_decoded, reduction='mean').detach().cpu().item()
            c_mses.append(c_mse)
        mse_per_sparsity.append(sum(c_mses)/len(c_mses))
    model.msk.p = prev_p
    model.msk.training = prev_training
    return mse_per_sparsity

def diffusion_fid_for_all_spars(model: nn.Module, test_dl: DataLoader, spars_params: dict, arch: dict, min: float=0.0, max: float=0.9, step: float=0.1, num_batches: int=10) -> list:
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
    model = model.to(DEVICE)
    model.eval()
    fid_per_sparsity = []
    prev_training = model.msk.training
    prev_p = model.msk.p
    # msk = ConditionMasking(arch, spars_params)
    for c_spars in tqdm(np.linspace(min, max, int((max-min)/step)+1)):
        fid.reset()
        # msk.p = c_spars
        model.msk.training = True
        model.msk.p = c_spars
        c_fids = []
        for i, (x, y) in enumerate(test_dl):
            if i == num_batches:
                break
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            with torch.no_grad():
                X_hat = model.sample((x.size(2)*8, x.size(3)*8), y)
            x_decoded = model.decode(x)
            fid.update(x_decoded, real=True)
            fid.update(X_hat, real=False)
            c_fids.append(fid.compute().detach().cpu().item())
        fid_per_sparsity.append(sum(c_fids)/len(c_fids))
    model.msk.p = prev_p
    model.msk.training = prev_training
    return fid_per_sparsity
            

def ensure_good_train_test_split(X, Y, arch, batch_size, seed, train_split):
    i = 0
    while True:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_split, shuffle=True, random_state=seed+i)
        missings = []
        for j, cond in enumerate(arch['cond_dims']):
            if cond in arch['numerical_cond']:
                missings.append(0)
            else:
                num_unique = np.unique(Y_train[:, j]).shape[0]
                missings.append(arch[cond] - num_unique)
        if sum(missings) == 0:
            print('Seed is: ', seed+i)
            break
        i += 1
    
    train_set = RefPointData(X_train, Y_train)
    test_set = RefPointData(X_test, Y_test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return X_train, X_test, Y_train, Y_test, train_set, test_set, train_loader, test_loader, seed+i

def car_dict_to_emb_cond_vector(c_cond):
    with open('./data/car_data/mappings_emb.json', 'r') as f:
        mappings = json.load(f)

    c_cond = np.array([
        mappings['manufacturer'][c_cond['manufacturer']],
        mappings['type'][c_cond['type']],
        mappings['class'][c_cond['class']],
        mappings['buzzwords'][c_cond['buzzword1']],
        mappings['buzzwords'][c_cond['buzzword2']],
        mappings['buzzwords'][c_cond['buzzword3']],
        c_cond['drag_coeff']
        ])
    c_cond = torch.tensor(c_cond, dtype=torch.float, device=DEVICE).unsqueeze(0)
    return c_cond