import numpy as np
import torch
import rasterio as rs 
import os
import sys
import pathlib
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import config

red_idx = config.S2_BANDS_NAMES_TO_USE.index("B4")
green_idx = config.S2_BANDS_NAMES_TO_USE.index("B3")
blue_idx = config.S2_BANDS_NAMES_TO_USE.index("B2")
nir_idx = config.S2_BANDS_NAMES_TO_USE.index("B8")


def _normalize_base(rgb: np.array):
    """
    Common function for the normalization of both tensor and paths images. 
    Args: 
        rgb (numpy.array) : an array with RGB bands stacked"""
    min_rgb, max_rgb = np.percentile(rgb, (2, 98), axis=(1, 2), keepdims=True)
    normalized_rgb = torch.tensor((rgb - min_rgb) / (max_rgb - min_rgb))

    return normalized_rgb

def normalize_rgb_tensor(image_tensor=None, red_idx = red_idx, green_idx = green_idx, blue_idx = blue_idx) -> torch.Tensor:
    """
    Normalizes the RGB channels of the input image to the range [0, 1] using the 2nd and 98th percentiles.
    
    Args:
        image (numpy.ndarray): Input image with shape (3, H, W) where the first dimension represents RGB channels.
    
    Outputs: 
        tensor without batch dim ready to plot with plt.imshow
    """
    if not isinstance(image_tensor, torch.Tensor):
        raise ValueError(f"{type(image_tensor)} is not a valid type. This function is used for tensors")
        return
    
    tensor_dim = image_tensor.dim()
    if tensor_dim != 3:
        raise ValueError(f"Image tensors should have 3 dimensions, your tensor has {tensor_dim} dimensions")
        return 
    
    red = image_tensor[red_idx]
    green = image_tensor[green_idx]
    blue = image_tensor[blue_idx]
    stacked = torch.stack((red, green, blue), dim=0)
    normalized_rgb = _normalize_base(stacked)
    return normalized_rgb

def normalize_rgb_path(image_path: str | pathlib.Path, red_idx = red_idx, green_idx = green_idx, blue_idx = blue_idx):
    with open(image_path, 'rb') as file:
        image_tensor = rs.open(file).read()
    
    red = image_tensor[red_idx,:,:]
    green = image_tensor[green_idx,:,:]
    blue = image_tensor[blue_idx,:,:]
    stacked = np.stack((red, green, blue))
    normalized_rgb = _normalize_base(stacked)
    return normalized_rgb

def plot_rgb_image(image: torch.Tensor | np.ndarray | str | pathlib.Path):
    if not isinstance(image, (torch.Tensor, np.ndarray, str, pathlib.Path)): 
        raise ValueError(f"Image should be a single tensor, array or path and is a {type(image)}")
    elif isinstance(image, (str, pathlib.Path)):
        rgb_tensor = normalize_rgb_path(image_path=image)
    elif isinstance(image, (torch.Tensor, np.ndarray)): 
        rgb_tensor = normalize_rgb_tensor(image_tensor=image)

    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.title('RGB S2 image')
    plt.imshow(rgb_tensor.cpu().permute(1,2,0))
    plt.show()

"""def subplot_rgb_list(image_list: list)
    if not isinstance(image_list, list): 
        raise ValueError(f"Image should be a single tensor, array or path and is a {type(image_list)}")

    n_images = len(image_list)
    if n_images > 9:
        print("---CONSOLE : your list is too big, the function will only plot the 9 first images---")
        n_images = 9
        image_list = image_list[:10]"""


if __name__ == '__main__':
    test_tensor = r"C:\Users\KenzoBounegta\SegSat\Sen1Floods11-Benchmark\dataset\sen1floods11_v1.1\sen1floods11_v1.1\data\S2L1CHand\Sri-Lanka_163406_S2Hand.tif"
    output = plot_rgb_image(test_tensor)
    


