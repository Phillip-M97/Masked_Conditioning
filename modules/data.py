import os
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch
torch.set_float32_matmul_precision('medium')  # pytorch lightning recommends this for faster training
from torch.utils.data import Dataset, DataLoader, random_split
from lightning import LightningDataModule
import random
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL
from math import ceil
from tqdm import tqdm
from safetensors.torch import save_file
from safetensors import safe_open

from .globals import DEVICE

# currently a copy of the version in utils.py, not nice but a quick fix for circular import
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


class RefPointData(Dataset):
    def __init__(self, X, Y) -> None:
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float, device=DEVICE)
        self.Y = torch.tensor(Y, dtype=torch.float, device=DEVICE)

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx) -> tuple:
        return self.X[idx], self.Y[idx]

class RefPointDataCarEmb(Dataset):
    def __init__(self, X: pd.DataFrame, Y: pd.DataFrame, num_buzzwords: int=3) -> None:
        super().__init__()
        self.X = torch.tensor(X.to_numpy(), dtype=torch.float, device=DEVICE)
        self.Y = Y
        self.num_buzzwords = num_buzzwords

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx) -> tuple:
        y = self.Y.iloc[idx]

        if 'buzzwords' in self.Y.columns:
            buzzwords = y['buzzwords'].split(';')
            buzzwords = [int(b) for b in buzzwords]

            if len(buzzwords) > self.num_buzzwords:
                buzzwords = random.sample(buzzwords, self.num_buzzwords)
            else:
                if len(buzzwords) == 0:
                    buzzwords.append(0)
                while len(buzzwords) < self.num_buzzwords:
                    buzzwords.extend(buzzwords)
                buzzwords = buzzwords[:self.num_buzzwords]

        else:
            buzzwords = []
            
        # expected order: manufacturer, type, class, buzz1, buzz2, buzz3, drag_coeff
        y_arr = np.array([y['manufacturer'], y['type'], y['class']] + buzzwords + [y['drag_coeff']], dtype=float)

        return self.X[idx], torch.tensor(y_arr, dtype=torch.float, device=DEVICE)

class BikeSketchDataset(Dataset):
    '''
    This Dataset class loads all images into torch tensors in main memory on initialization.
    This is fast but consumes a lot of RAM (all images are always in RAM).
    For larger image datasets (or high res images) it might be necessary to adapt the implementation s.t. images are loaded into main memory JIT.
    '''

    def __init__(self, images_path: str, Y: pd.DataFrame, image_size: int=256, keep_cpu: bool=False, mode:str='L', use_norm: bool=False, use_simple_norm: bool=False, data_mean: float=0.9466, data_std: float=0.1910) -> None:
        super(BikeSketchDataset, self).__init__()
        self.images_fnames = sorted(os.listdir(images_path))
        
        unneeded_columns = [x for x in Y.columns if 'x_' in x or 'y_' in x or 'Bike index' in x or 'diameter' in x] + ['Unnamed: 0']
        Y_cond = Y.drop(columns=unneeded_columns)
        Y_cond.columns = ['BikeStyle', 'TeethChain', 'BottleSeatTube', 'BottleDownTube', 'FrameSize', 'RimStyleFront', 'RimStyleRear', 'ForkType']

        transform = [
            transforms.ToTensor(),
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST_EXACT),
        ]

        if use_norm:
            transform.append(transforms.Normalize(mean=[data_mean], std=[data_std]))
        elif use_simple_norm:
            transform.append(transforms.Lambda(lambda x: x*2-1))

        transform = transforms.Compose(transform)
        
        ys = []
        xs = []
        print('\n-- loading images --\n')
        # TODO: much more efficient if we compute once per dataset and then store a SafeTensor
        for fname in tqdm(self.images_fnames):
            bike_no = fname.split('.')[0]
            c_y = Y[Y['Bike index'] == int(bike_no)]
            c_y = c_y.drop(columns=unneeded_columns)
            c_y.columns = ['BikeStyle', 'TeethChain', 'BottleSeatTube', 'BottleDownTube', 'FrameSize', 'RimStyleFront', 'RimStyleRear', 'ForkType']
            c_y_arr = get_conditioning_vector_embedding_biked(c_y.iloc[0].to_dict(), Y_cond)
            ys.append(c_y_arr)

            image = Image.open(os.path.join(images_path, fname)).convert(mode)
            # image = Image.new(mode=mode, size=(image_size, image_size), color=(255, 255, 255))
            image_t = transform(image)
            xs.append(image_t)
        self.Y = torch.tensor(np.stack(ys, axis=0)).to(torch.float)
        self.X = torch.stack(xs)
        print('\n-- all images loaded from disk --\n')
        if not keep_cpu:
            self.Y = self.Y.to(DEVICE)
            self.X = self.X.to(DEVICE)

    def __len__(self) -> int:
        return self.Y.size(0)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.Y[idx]
    
class BikedDataModule(LightningDataModule):

    def __init__(self, images_dir: str, conditions_df_path: str, mode: str='RGB', batch_size: int=32, image_size: int=512, train_split: float=0.1, num_workers: int=16) -> None:
        super(BikedDataModule, self).__init__()
        self.images_dir = images_dir
        self.conditions_df_path = conditions_df_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.train_split = train_split
        self.mode = mode
        self.num_workers = num_workers
        
    def setup(self, stage: str) -> None:
        df_y = pd.read_csv(self.conditions_df_path)
        full_data = BikeSketchDataset(self.images_dir, df_y, self.image_size, mode=self.mode, keep_cpu=True, use_norm=False, use_simple_norm=True)
        
        train_size = int(self.train_split*len(full_data))
        val_size = len(full_data) - train_size
        self.train_data, self.val_data = random_split(full_data, [train_size, val_size])
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class LatentBikeSketchDataset(BikeSketchDataset):
    '''
    This Dataset class loads all images into torch tensors in main memory on initialization.
    This is fast but consumes a lot of RAM (all images are always in RAM).
    For larger image datasets (or high res images) it might be necessary to adapt the implementation s.t. images are loaded into main memory JIT.
    '''

    def __init__(self, images_path: str, Y: pd.DataFrame, vae: AutoencoderKL, image_size: int=256, keep_cpu: bool=False, device: str='cuda', mode:str='L', use_norm: bool=False, use_simple_norm: bool=False, data_mean: float=0.9466, data_std: float=0.1910) -> None:
        super(LatentBikeSketchDataset, self).__init__(images_path, Y, image_size, keep_cpu, mode, use_norm, use_simple_norm, data_mean, data_std)
        latent_xs = None
        vae.eval()
        vae.to(device)
        # TODO: much more efficient if we compute once per dataset and then store a SafeTensor
        print('\n-- computing latents --\n')
        with torch.no_grad():
            for i in tqdm(range(ceil(self.X.size(0)/16))):
                c_x = self.X[i*16:min(self.X.size(0), (i+1)*16)]
                c_x = c_x.to(device)
                c_latent = vae.encode(c_x)
                c_latent = c_latent.latent_dist.sample()
                c_latent = 0.18215 * c_latent
                if keep_cpu:
                    c_latent = c_latent.cpu()
                if latent_xs is None:
                    latent_xs = c_latent
                else:
                    latent_xs = torch.cat((latent_xs, c_latent), dim=0)
        self.X = latent_xs
        print('\n-- precomputed latents stored --\n')

class LatentBikedDataModule(BikedDataModule):

    def __init__(self, images_dir: str, conditions_df_path: str, vae: AutoencoderKL, device: str, mode: str='RGB', batch_size: int=32, image_size: int=512, train_split: float=0.1, num_workers: int=16) -> None:
        super(LatentBikedDataModule, self).__init__(images_dir, conditions_df_path, mode, batch_size, image_size, train_split, num_workers)
        self.vae = vae
        self.device = device

    def setup(self, stage: str) -> None:
        df_y = pd.read_csv(self.conditions_df_path)
        full_data = LatentBikeSketchDataset(self.images_dir, df_y, self.vae, self.image_size, mode=self.mode, keep_cpu=True, device=self.device, use_norm=False, use_simple_norm=True)
        
        train_size = int(self.train_split*len(full_data))
        val_size = len(full_data) - train_size
        self.train_data, self.val_data = random_split(full_data, [train_size, val_size])

# -- DVM Car Dataset --

def get_info(img_path: str, df_images: pd.DataFrame, df_price: pd.DataFrame, df_ad: pd.DataFrame):
    image_name = img_path.split('/')[-1]
    img_row = df_images[df_images['Image_name'] == image_name]
    try:
        # view_point = img_row.iloc[0]['Predicted_viewpoint']  # Removed viewpoint because we are using quality checked front views
        clean_img_name = image_name.split('.')[0]
        splits = clean_img_name.split('$$')
        brand, model, year, color = splits[0], splits[1], splits[2], splits[3]
        year = int(year)

        price_row = df_price[(df_price['Maker'] == brand) & (df_price['Genmodel'] == model) & (df_price['Year'] == int(year))]
        entry_price = price_row.iloc[0]['Entry_price'] if len(price_row) > 0 else -1

        df_ad_rows = df_ad[(df_ad['Maker'] == brand) & (df_ad[' Genmodel'] == model) & (df_ad['Adv_year'] == int(year))]
        car_type = df_ad_rows[df_ad_rows['Bodytype'].notna()]['Bodytype'].iloc[0] if len(df_ad_rows[df_ad_rows['Bodytype'].notna()]) > 0 else 'unknown'
        length = df_ad_rows[df_ad_rows['Length'].notna()]['Length'].iloc[0] if len(df_ad_rows[df_ad_rows['Length'].notna()]) > 0 else -1
        width = df_ad_rows[df_ad_rows['Width'].notna()]['Width'].iloc[0] if len(df_ad_rows[df_ad_rows['Width'].notna()]) > 0 else -1
        height = df_ad_rows[df_ad_rows['Height'].notna()]['Height'].iloc[0] if len(df_ad_rows[df_ad_rows['Height'].notna()]) > 0 else -1
        num_doors = df_ad_rows[df_ad_rows['Door_num'].notna()]['Door_num'].iloc[0] if len(df_ad_rows[df_ad_rows['Door_num'].notna()]) > 0 else -1
        num_seats = df_ad_rows[df_ad_rows['Seat_num'].notna()]['Seat_num'].iloc[0] if len(df_ad_rows[df_ad_rows['Seat_num'].notna()]) > 0 else -1
        power = df_ad_rows[df_ad_rows['Engine_power'].notna()]['Engine_power'].iloc[0] if len(df_ad_rows[df_ad_rows['Engine_power'].notna()]) > 0 else -1
        mpg = df_ad_rows[df_ad_rows['Average_mpg'].notna()]['Average_mpg'].iloc[0] if len(df_ad_rows[df_ad_rows['Average_mpg'].notna()]) > 0 else '-1 mpg'
        mpg = float(mpg.replace(' mpg', ''))
        top_speed = df_ad_rows[df_ad_rows['Top_speed'].notna()]['Top_speed'].iloc[0] if len(df_ad_rows[df_ad_rows['Top_speed'].notna()]) > 0 else '-1 mph'
        top_speed = float(top_speed.replace(' mph', ''))

        return {
            # 'view_point': view_point,
            'brand': brand,
            'model': model,
            'type': car_type,
            'year': year,
            'color': color,
            'entry_price': entry_price,
            'length': length,
            'width': width,
            'height': height,
            'num_doors': num_doors,
            'num_seats': num_seats,
            'power': power,
            'mpg': mpg,
            'top_speed': top_speed
        }
    except IndexError:
        print('File not found in tables: ', img_path)
        return None 

def encode_value(value, column_name: str, df: pd.DataFrame):
    all_values = df[column_name].unique()
    all_values.sort()
    all_values_list = all_values.tolist()
    try:
        return all_values_list.index(value)
    except ValueError:
        return -1

def normalize_value(value: float, column_name: str, df: pd.DataFrame) -> float:
    if value == -1:
        return -1
    values = df[column_name]
    if values.dtype == 'O' or values.dtype == 'str':
        values = values[~values.isna()]
        values = values.str.extract(r'(\d+(?:\.\d+)?)').astype(float)
        values = values[values.columns[0]]
    min_value = values.min()
    max_value = values.max()
    values = values.fillna(min_value)
    normalized_value = (value - min_value) / (max_value - min_value)
    return normalized_value

def encode_values(condition: dict, df_images: pd.DataFrame, df_basic: pd.DataFrame, df_ad: pd.DataFrame, df_price: pd.DataFrame) -> np.array:
    encoded = [
        # encode_value(condition['view_point'], 'Predicted_viewpoint', df_images),  # removed viewpoint because we are using quality checked front views
        encode_value(condition['brand'], 'Automaker', df_basic),
        encode_value(condition['model'], 'Genmodel', df_basic),
        encode_value(condition['type'], 'Bodytype', df_ad[~df_ad['Bodytype'].isna()]),
        normalize_value(condition['year'], 'Year', df_images),
        encode_value(condition['color'], 'Color', df_images),
        normalize_value(condition['entry_price'], 'Entry_price', df_price),
        normalize_value(condition['length'], 'Length', df_ad),
        normalize_value(condition['width'], 'Width', df_ad),
        normalize_value(condition['height'], 'Height', df_ad),
        normalize_value(condition['num_doors'], 'Door_num', df_ad),
        normalize_value(condition['num_seats'], 'Seat_num', df_ad),
        normalize_value(condition['power'], 'Engine_power', df_ad),
        normalize_value(condition['mpg'], 'Average_mpg', df_ad),
        normalize_value(condition['top_speed'], 'Top_speed', df_ad)
    ]
    return np.array(encoded).astype(np.float32)

class LatentDVMDataset(Dataset):

    def __init__(self, images_path: str, conditions_path: str, vae: AutoencoderKL, image_size: int, cache: bool=True, save_pth: str='./tensor_cache', device: str='cuda') -> None:
        super(Dataset, self).__init__()

        self.device = device

        pth = os.path.join(save_pth, f"{images_path[1:].replace('/', '-')}.safetensors")
        if os.path.exists(pth) and cache:
            print('Using Cached Latents')
            tensors = {}
            with safe_open(pth, framework='pt', device='cpu') as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
            self.X = tensors['X']
            self.Y = tensors['Y']
            return
        
        os.makedirs(save_pth, exist_ok=True)

        def extract_from_name(image_name, index: int):
            try:
                parts = image_name.split('$$')
                if len(parts) >= 4:
                    return parts[index]
                else:
                    return None
            except AttributeError:
                return 'unknown'
        df_images = pd.read_csv(os.path.join(conditions_path, 'Image_table.csv'))
        df_images['Color'] = df_images['Image_name'].apply(lambda x: extract_from_name(x, 3))
        df_images['Year'] = df_images['Image_name'].apply(lambda x: int(extract_from_name(x, 4)))
        df_ad = pd.read_csv(os.path.join(conditions_path, 'Ad_table (extra).csv'))
        df_price = pd.read_csv(os.path.join(conditions_path, 'Price_table.csv'))
        df_basic = pd.read_csv(os.path.join(conditions_path, 'Basic_table.csv'))

        transform = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x*2-1),
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST_EXACT)
        ]
        transform = transforms.Compose(transform)

        """ vae.eval()
        vae = vae.to(device) """
        self.X = []
        self.Y = []
        brand_dirs = sorted(os.listdir(images_path))

        print('-- Calculating Latents --')

        for b_d in tqdm(brand_dirs, desc='loading (brands)'):
            c_b_pth = os.path.join(images_path, b_d)
            model_dirs = sorted(os.listdir(c_b_pth))
            for m_d in model_dirs:
                c_m_pth = os.path.join(c_b_pth, m_d)
                year_dirs = sorted(os.listdir(c_m_pth))
                for y_d in year_dirs:
                    c_y_pth = os.path.join(c_m_pth, y_d)
                    color_dirs = sorted(os.listdir(c_y_pth))
                    for c_d in color_dirs:
                        c_c_pth = os.path.join(c_y_pth, c_d)
                        img_names = sorted(os.listdir(c_c_pth))
                        for img_fname in img_names:
                            c_img_pth = os.path.join(c_c_pth, img_fname)
                            c_img = Image.open(c_img_pth)
                            """ c_img_t = transform(c_img).unsqueeze(0).to(self.device)
                            c_latent = vae.encode(c_img_t).latent_dist.sample() * 0.18215
                            self.X.append(c_latent.squeeze().detach().cpu()) """
                            c_img_t = transform(c_img)

                            c_info = get_info(c_img_pth, df_images, df_price, df_ad)
                            if c_info is None:  # if index error occurs
                                continue
                            c_info_enc = encode_values(c_info, df_images, df_basic, df_ad, df_price)
                            self.Y.append(torch.tensor(c_info_enc))
                            self.X.append(c_img_t)
        self.X = torch.stack(self.X)
        self.Y = torch.stack(self.Y)

        vae.eval()
        vae = vae.to(device)
        latents = None
        bs = 20
        with torch.no_grad():
            for i in tqdm(range(ceil(len(self.X)/bs)), desc='latent batches'):
                c_X = self.X[i*bs:min(len(self.X),(i+1)*bs)]
                c_X = c_X.to(device)
                c_latent = vae.encode(c_X).latent_dist.sample() * 0.18215
                c_latent = c_latent.detach().cpu()
                latents = c_latent if latents is None else torch.cat((latents, c_latent), dim=0)
        self.X = latents

        vae = vae.to('cpu')
        print('-- latents computed --')
        # store data as safetensor
        if cache:
            tensors = {
                "X": self.X,
                "Y": self.Y
            }
            save_file(tensors, pth)
    
    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.Y[idx]

class LatentDVMDataModule(LightningDataModule):

    def __init__(self, images_dir: str, conditions_df_path: str, vae: AutoencoderKL, image_size: int, batch_size: int=32, train_split: float=0.1, num_workers: int=16, device: str='cuda', cache: bool=True) -> None:
        super(LatentDVMDataModule, self).__init__()
        self.images_dir = images_dir
        self.conditions_df_path = conditions_df_path
        self.batch_size = batch_size
        self.train_split = train_split
        self.num_workers = num_workers
        self.vae = vae
        self.device = device
        self.cache = cache
        self.image_size=image_size
        
    def setup(self, stage: str) -> None:
        full_data = LatentDVMDataset(self.images_dir, self.conditions_df_path, self.vae, image_size=self.image_size, cache=self.cache, device=self.device)
        train_size = int(self.train_split*len(full_data))
        val_size = len(full_data) - train_size
        self.train_data, self.val_data = random_split(full_data, [train_size, val_size])
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

# -- wrapper --

def get_data_module(module_type: str, images_dir: str, conditions_df_path: str, vae: AutoencoderKL=None, device: str=None, batch_size: str=None, image_size: str=None, train_split: float=None, cache: bool=True) -> LightningDataModule:
    if module_type == 'biked':
        return BikedDataModule(
            images_dir=images_dir,
            conditions_df_path=conditions_df_path,
            batch_size=batch_size,
            image_size=image_size,
            train_split=train_split
        )
    if module_type == 'biked_latent':
        return LatentBikedDataModule(
            images_dir=images_dir,
            conditions_df_path=conditions_df_path,
            vae=vae,
            device=device,
            batch_size=batch_size,
            image_size=image_size,
            train_split=train_split
        )
    if module_type == 'dvm_latent':
        return LatentDVMDataModule(
            images_dir=images_dir,
            conditions_df_path=conditions_df_path,
            vae=vae,
            image_size=image_size,
            batch_size=batch_size,
            train_split=train_split,
            device=device,
            cache=cache
        )
    raise NotImplementedError('The module type you have chosen is not available, please choose one of "biked", "biked_latent", "dvm_latent"')
