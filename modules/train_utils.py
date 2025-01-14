from .globals import DEVICE
from .mc_sa_alae import mcSAALAE, update_step_mcSAALAE, get_alae_losses
from .utils import evaluate_for_all_spars
from .emb_mcvae import ConditionMasking
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
from copy import deepcopy


'''
The methods below are used to train an mc-SA-ALAE model.
'''
# in each epoch we iterate through the training dataloader and update the model for each batch
def alae_epoch(model: mcSAALAE, train_loader: DataLoader, noise_dim: int, optimizer_encoder: optim.Optimizer, optimizer_discriminator: optim.Optimizer, optimizer_mapper: optim.Optimizer, optimizer_generator: optim.Optimizer, gamma: float=20.) -> list:
    model.train()
    full_loss, disc_loss, gen_loss, cons_loss, full_mse = 0, 0, 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        n = torch.randn((x.size(0), noise_dim)).to(DEVICE)
        discriminator_loss, generator_loss, latent_consistency_loss, mse = update_step_mcSAALAE(model, x, y, optimizer_encoder, optimizer_discriminator, optimizer_mapper, optimizer_generator, n, gamma)
        loss = discriminator_loss + generator_loss + latent_consistency_loss

        full_loss += loss.detach().cpu().item() * len(x)
        disc_loss += discriminator_loss.detach().cpu().item() * len(x)
        gen_loss += generator_loss.detach().cpu().item() * len(x)
        cons_loss += latent_consistency_loss.cpu().detach().item() * len(x)
        full_mse += mse.detach().cpu().item() * len(x)
    
    full_loss = (full_loss/len(train_loader.dataset))
    disc_loss = (disc_loss/len(train_loader.dataset))
    gen_loss = (gen_loss/len(train_loader.dataset))
    cons_loss = (cons_loss/len(train_loader.dataset))
    full_mse = (full_mse/len(train_loader.dataset))

    return [full_loss, disc_loss, gen_loss, cons_loss, full_mse]

# to validate the training we iterate through the validation dataloader and average losses
def alae_val_epochs(model: mcSAALAE, val_loader: DataLoader, noise_dim: int, gamma: float=20.) -> list:
    model.eval()
    val_loss, val_disc_loss, val_gen_loss, val_cons_loss, val_mse = 0, 0, 0, 0, 0
    
    for x, y in val_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        n = torch.randn((x.size(0), noise_dim)).to(DEVICE)
        discriminator_loss, generator_loss, latent_consistency_loss, mse = get_alae_losses(model, x, y, n, gamma)
        loss = discriminator_loss + generator_loss + latent_consistency_loss

        val_loss += loss.detach().cpu().item() * len(x)
        val_disc_loss += discriminator_loss.detach().cpu().item() * len(x)
        val_gen_loss += generator_loss.detach().cpu().item() * len(x)
        val_cons_loss += latent_consistency_loss.detach().cpu().item() * len(x)
        val_mse += mse.detach().cpu().item() * len(x)
    
    val_loss = (val_loss/len(val_loader.dataset))
    val_disc_loss = (val_disc_loss/len(val_loader.dataset))
    val_gen_loss = (val_gen_loss/len(val_loader.dataset))
    val_cons_loss = (val_cons_loss/len(val_loader.dataset))
    val_mse = (val_mse/len(val_loader.dataset))

    return [val_loss, val_disc_loss, val_gen_loss, val_cons_loss, val_mse]

# for each epoch we iterate the training and validation steps, check the early stopping criterion and print some results
def train_alae(model: mcSAALAE, train_loader: DataLoader, val_loader: DataLoader, train_params: dict, spars_params: dict, arch: dict, verbose: int=2, early_stopping: bool=False, es_eval_method=None, es_patience: int=3):
    # using AdamW instead of Adam for superior generalization performance, betas are taken from ALAE and SA-ALAE paper
    optimizer_encoder = optim.AdamW(model.encoder.parameters(), lr=train_params['encoder_lr'], betas=(0.0, 0.99))
    # idea: SGD for discriminator
    optimizer_discriminator = optim.AdamW(model.discriminator.parameters(), lr=train_params['disc_lr'], betas=(0.0, 0.99))
    optimizer_mapper = optim.AdamW(model.mapper.parameters(), lr=train_params['mapper_lr'], betas=(0.0, 0.99))
    optimizer_generator = optim.AdamW(model.generator.parameters(), lr=train_params['generator_lr'], betas=(0.0, 0.99))

    if es_eval_method is None:
        es_eval_method = lambda c_model: mse_criterion(c_model, val_loader, arch, spars_params)
    else:
        es_eval_method = lambda c_model: es_eval_method(c_model, val_loader)

    spars_scheduler = spars_params['sparsity_scheduler'](model, arch, train_params, spars_params)

    spars_hist = spars_scheduler.get_sparsity()
    train_hist, val_hist, es_hist = [], [], []
    last_es_eval = 1e12
    num_epochs_wo_improvement = 0
    for epoch in range(1, train_params['epochs'] + 1):
        # training epoch
        train_losses = alae_epoch(model, train_loader, arch['noise_dim'], optimizer_encoder, optimizer_discriminator, optimizer_mapper, optimizer_generator, train_params['encoder_reg_weight'])
        train_hist.append(train_losses)
        
        # validation epoch
        val_losses = alae_val_epochs(model, val_loader, arch['noise_dim'], train_params['encoder_reg_weight'])
        val_hist.append(val_losses)

        spars_scheduler.step()
        
        # printing of results in current epoch
        if verbose >= 3 or (verbose >= 2 and epoch % 10 == 0):
            print(
                f'Epoch: {epoch:5}  ' +
                f'Summed Loss: {train_losses[0]:5.3f}  ' +
                f'MSE: {train_losses[-1]:5.3f}  ' +
                f'Discriminator Loss: {train_losses[1]:5.3f}  ' +
                f'Generator Loss: {train_losses[2]:5.3f}  ' +
                f'Latent Loss: {train_losses[3]:5.3f}  ' +
                f'Val Summed Loss: {val_losses[0]:5.3f}  ' +
                f'Val MSE: {val_losses[-1]:5.3f}  ' +
                f'Val Discriminator Loss: {val_losses[1]:5.3f}  ' +
                f'Val Generator Loss: {val_losses[2]:5.3f}  ' +
                f'Val Latent Loss: {val_losses[3]:5.3f}  ' +
                f'Spars: {spars_scheduler.get_last_sparsity():.5f}'
            )
        
        # early stopping as described in the SA-ALAE paper but they used FID, we (by default) use the mean MSE over sparsities [0.0, 0.9]
        if early_stopping:
            es_eval = es_eval_method(model)
            es_hist.append(es_eval)
            if es_eval < last_es_eval:
                last_es_eval = es_eval
                num_epochs_wo_improvement = 0
            else:
                num_epochs_wo_improvement += 1
                if num_epochs_wo_improvement >= es_patience:
                    if verbose >= 1:
                        print(f'--- EARLY STOPPING in epoch {epoch} after {num_epochs_wo_improvement} epochs without improvement ---')
                    break
        else:
            es_hist.append(0.)
    
    # final printing
    train_hist = np.array(train_hist)
    val_hist = np.array(val_hist)
    spars_hist = np.array(spars_hist)
    es_hist = np.array(es_hist)
    # train_hist, val_hist, spars_hist, es_hist = np.array(train_hist), np.array(val_hist), np.array(spars_hist), np.array(es_hist)
    
    if verbose >= 0:
        print(
            '\nCurrent Loss\t>>> ' +
            f'Summed Loss: {train_hist[-1, 0]:5.3f}  ' +
            f'MSE: {train_hist[-1, -1]:5.3f}  ' +
            f'Discriminator Loss: {train_hist[-1, 1]:5.3f}  ' +
            f'Generator Loss: {train_hist[-1, 2]:5.3f}  ' +
            f'Latent Loss: {train_hist[-1, 3]:5.3f}  ' +
            f'Val Summed Loss: {val_hist[-1, 0]:5.3f}  ' +
            f'Val MSE: {val_hist[-1, -1]:5.3f}  ' +
            f'Val Discriminator Loss: {val_hist[-1, 1]:5.3f}  ' +
            f'Val Generator Loss: {val_hist[-1, 2]:5.3f}  ' +
            f'Val Latent Loss: {val_hist[-1, 3]:5.3f}  '
        )
    return train_hist, val_hist, spars_hist, es_hist

def plot_alae_training(train_hist: np.array, val_hist: np.array, spars_hist: np.array, es_hist: np.array, suptitle: str=None) -> None:
    plt.figure(figsize=(12, 8))

    if suptitle:
        plt.suptitle(suptitle)

    plt.subplot(2, 2, 1)
    plt.title('Summed Loss')
    plt.plot(list(range(len(train_hist))), train_hist[:, 0], label='train')
    plt.plot(list(range(len(val_hist))), val_hist[:, 0], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Summed Loss Value')
    plt.legend(loc='upper right')

    plt.subplot(2, 2, 2)
    plt.title('ALAE losses')
    plt.plot(list(range(len(train_hist))), train_hist[:, 1], label='train discriminator')
    plt.plot(list(range(len(train_hist))), train_hist[:, 2], label='train generator')
    plt.plot(list(range(len(train_hist))), train_hist[:, 3], label='train latent consistency')
    plt.plot(list(range(len(val_hist))), val_hist[:, 1], label='val discriminator')
    plt.plot(list(range(len(val_hist))), val_hist[:, 2], label='val generator')
    plt.plot(list(range(len(val_hist))), val_hist[:, 3], label='val latent consistency')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend(loc='upper right')
    

    plt.subplot(2, 2, 3)
    plt.title('Mean MSE per Sparsity')
    plt.plot(list(range(len(es_hist))), es_hist)
    plt.xlabel('Epoch')
    plt.ylabel('Mean MSE over Sparsities [0.0, 0.9]')

    plt.subplot(2, 2, 4)
    plt.title('Sparsity')
    plt.plot(list(range(len(spars_hist))), spars_hist)
    plt.xlabel('Epoch')
    plt.ylabel('Zero-Out Probabilty')

    plt.tight_layout()

    plt.show()

def fid_criterion(model: mcSAALAE, test_loader: DataLoader, num_samples=100, feature=64, normalize=True) -> float:
    random_loader = DataLoader(deepcopy(test_loader.dataset), batch_size=num_samples, shuffle=True)
    x, y = next(iter(random_loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    x_hat = model.generate_new(y)
    return fid(x, x_hat, feature=feature, normalize=normalize)


def fid(real_imgs, gen_imgs, feature=64, normalize=True) -> float:
    if real_imgs.size(1) < 3:
        real_imgs = real_imgs.repeat(1, 4-real_imgs.size(1), 1, 1)
        gen_imgs = gen_imgs.repeat(1, 4-gen_imgs.size(1), 1, 1)
    fid_metric = FrechetInceptionDistance(feature=feature, normalize=normalize).to(real_imgs.device)

    fid_metric.update(real_imgs, real=True)
    fid_metric.update(gen_imgs, real=False)

    return fid_metric.compute().detach().cpu().item()

def get_mse_per_spars(model: mcSAALAE, loader: DataLoader, arch: dict, spars_params: dict, num_batches: int=100000) -> list:
    x_all, y_all = [], []
    for i, data in enumerate(loader):
        if i > num_batches:
            break
        x, y = data
        x_all.append(x)
        y_all.append(y)
    x, y = torch.cat(x_all, dim=0), torch.cat(y_all, dim=0)
    x, y = x.to(DEVICE), y.to(DEVICE)
    msk = ConditionMasking(arch, spars_params)

    y_per_spars = []
    for c_spars in np.linspace(0.0, 0.9, int((1-0.1)/0.1)+1):
        msk.p = c_spars
        y_msk = msk(y.clone())
        y_per_spars.append(y_msk)
    
    model.eval()
    mse_per_spars = []
    for c_y in y_per_spars:
        n = torch.randn((c_y.size(0), model.noise_dim)).to(c_y.device)
        c_y_emb = model.mapper.encode(model.mapper.embed_y(c_y))
        x_hat = model.generator(c_y_emb, n)
        c_mse = ((x - x_hat)**2).mean()
        mse_per_spars.append(c_mse.detach().cpu().item())
    return mse_per_spars

def mse_criterion(model: mcSAALAE, test_loader: DataLoader, arch: dict, spars_params: dict, num_samples: int=100) -> float:
    es_loader = DataLoader(deepcopy(test_loader.dataset), batch_size=num_samples, shuffle=True)
    mse_per_spars = get_mse_per_spars(model, es_loader, arch, spars_params, num_batches=1)
    return sum(mse_per_spars)/len(mse_per_spars)