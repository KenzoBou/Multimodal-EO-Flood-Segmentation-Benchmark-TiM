import torch
from torch.utils.data import Dataset
from pathlib import Path
import rasterio
import numpy as np
import config
from torchvision import transforms
import random
import scipy

class Sen1FloodsDataset(Dataset):
    def __init__(self,
                 root_dir,
                 bands = list(range(1,14)),
                 modalities = ['S2', 'S1'],
                 split:str = 'train',
                 normalize = True):
        super().__init__()
        self.bands = tuple(bands) if type(bands) == list else bands
        self.root_dir = root_dir
        self.split = split
        self.modalities = modalities
        self.normalize = normalize
        S2_path = self.root_dir / "S2L1CHand"
        self.s2_img_list = [f for f in S2_path.glob('*.tif') if not f.name.startswith('._')]
        rng = random.Random(config.RANDOM_SEED)
        rng.shuffle(self.s2_img_list)
        dataset_size = len(self.s2_img_list)
        if self.split == 'train' :
            self.s2_img_list = self.s2_img_list[:int(dataset_size*0.7)] 
        elif self.split=='val' :
            self.s2_img_list = self.s2_img_list[int(dataset_size*0.7):int(dataset_size*0.8)]
        elif self.split=='test':
            self.s2_img_list = self.s2_img_list[int(dataset_size*0.8):]

    def __len__(self):
        return len(self.s2_img_list)
    
    @staticmethod
    def interpolate_nans(image:np.array):
        band_copy = np.copy(image)
        valid_mask = ~np.isnan(band_copy)
        invalid_mask = np.isnan(band_copy)
        if not invalid_mask.any():
            return band_copy
        elif not valid_mask.any(): #if there are only nans, we print an error message and output 0
            band_copy[invalid_mask] = 0
            return band_copy
        else: 
            valid_coords = np.array(np.where(valid_mask)).T
            invalid_coords = np.array(np.where(invalid_mask)).T
            valid_values = band_copy[valid_mask]
            interpolated_values = scipy.interpolate.griddata(valid_coords, valid_values, invalid_coords, method='nearest')
            band_copy[invalid_mask] = interpolated_values
            return band_copy

    
    def __getitem__(self, idx):
        s2_img = self.s2_img_list[idx]
        s2_img_stem = s2_img.stem.replace("S2Hand", '')
        S1_path = self.root_dir / "S1GRDHand" / f"{s2_img_stem}S1Hand.tif"
        mask_path = self.root_dir / "LabelHand" / f"{s2_img_stem}LabelHand.tif"
        if 'S2' in self.modalities:
            with rasterio.open(self.s2_img_list[idx]) as sample:
                s2_bands = torch.from_numpy(sample.read(self.bands)).float()

        if 'S1' in self.modalities:
            try :
                with rasterio.open(S1_path) as sample:
                    s1_bands = sample.read()
                    if np.isnan(s1_bands).any():
                        s1_bands = self.interpolate_nans(s1_bands)
                    s1_bands = torch.from_numpy(s1_bands).float()
            except FileNotFoundError:
                print(f"S1 file not found for {s2_img_stem}")
                self.modalities.remove('S1')
  
        with rasterio.open(mask_path) as sample:
            mask = sample.read()
            # we will remove the -1 from the mask
            mask[mask == -1] = 255  # Assuming -1 is the background class
            mask_tensor = torch.from_numpy(mask).long()  # Convert mask to long tensor for segmentation tasks

        image_tensor = torch.concat((s2_bands, s1_bands), dim=0).float() if 'S1' in self.modalities else s2_bands.float()
        
        if self.normalize:
            index_modalities = [band - 1 for band in self.bands]+ [-2,-1] if 'S1' in self.modalities else [band - 1 for band in self.bands]
            mean = np.array(config.MEAN_S2_BANDS)[index_modalities]
            std = np.array(config.STD_S2_BANDS)[index_modalities]
            image_tensor = transforms.Normalize(mean=mean, std=std)(image_tensor)
        
        return image_tensor, mask_tensor
