# Masked Conditioning
In this folder you may find code related to the *Masked Conditioning* project. Roughly speaking, two different models are implemented: mcVAE and mcLDM.

## mcVAE
To summarize an mcVAE works by training a standard VAE and concatenating an embedded conditioning vector to the latent representation before applying the decoder. Further, the conditioning information is masked randomly s.t. only a random subset of all conditions is available for the model at any time during training. The masking probability i.e. how many conditions are removed at any point in time is changed during training using a sparsity scheduler. This training regime enables the model to work with an arbitrary amount of conditions at inference time, allowing the user to specify only a sparse subset of conditions instead of the full conditioning vector and still obtain high quality results.

To train an mcVAE on parametric/vector data use the `train_mcvae.py` script as follows:
```bash
python train_mcvae.py --csv_path <path to csv with ref points and conditions> --name <name of your training run>
```

To train a convolutional mcVAE on image data use the `train_conv_mcvae.py` script as follows:
```bash
python train_conv_mcvae.py --images_path <path to your images> --csv_path <path to csv with conditions> --name <name of your training run>
```

Note that both *training scripts* are *currently only applicable to the Biked dataset*. To train mcVAE models on your own data you will have to implement appropriate preprocessing, potentially a new Dataset class, and modify the `get_configuration_biked` functions. As an example you may refer to `./_notebooks/emb_mcvae_with_cars.ipynb`.

## mcLDM
To improve the quality of sampled images, mcLDM extends the approach developed for mcVAE to latent diffusion models. This means we apply a iterative denoising diffusion process in the latent space of a VAE model conditioned on our masked conditioning vectors. We use a pretrained high-quality VAE as the first stage. We observed that the SD2.1 VAE worked best for our use case (SD1.5 struggled too much with fine grained textures found in technical images and SDXL produced NaN values for some images). With mcVAE the condition was only concatenated to the latent representation once. Differing from this implementation, we follow Stable Diffusion and integrate the conditioning information at each resolution level of the Unet downward and upward path. However, to reduce parameter counts, we do not apply cross-attention as in large Text2Image models but resort to simple concatenation and convolution.

To train an mcLDM use the `train_mcldm.py` script as follows:
```bash
python train_mcldm.py --cfg_name <name of your config file without .yaml>
```

The full CLI is explained here:
| Argument | Description | Example |
|---|---|---|
| --cfg_name | Name of the configuration file to use without .yaml ending. File must be in ./configs | --cfg_name mcldm_biked |
| --name | Name of the current training run. Used to identify log files. | --name biked_500k |
| --use_wandb | If set the weights and biases online logging is used. | --use_wannd |
| --run_id | The weights and biases run id to use. Can be set to continue an earlier training. | --run_id 5wk87bmq |
| --ckpt_path | A checkpoint file to use to resume training from. | --ckpt_path /logs/dvm_375k_2024-07-05-11-16-08/checkpoints/last.ckpt |

Please refer to the provided configuration files as examples on how to configure models.

## Repository Structure
| Path | Description |
|---|---|
| train_*.py | Training scripts for mcLDM and mcVAE. |
| ./_notebooks | Jupyter notebooks with code for some dataset analysis, experiments with different types of architecture, experiments with different datasets, dataset preprocessing, some evaluation. |
| ./configs | Configuration files defining architecture and training hyperparameters for (latent) diffusion models. |
| ./experiments | Jupyter notebooks for specific experiments. Contains notebooks for SHAP studies, explainability experiments, hyperparameter tuning, sparsity scheduler parameter tuning, experiments with dataset size, and some other experiments. |
| ./modules | Reusable code for the project. Contains utilities, sparsity scheduler implementation, implementation of all models, training scripts, evaluation scripts, loss functions, and global parameters. |
| ./modules/diffusion_components | Implementation of components necessary for a diffusion mode. Includes beta schedule, time embedding layer, the time conditional Unet, Up and Down blocks of the Unet backbone, and the modified ResBlocks. |
