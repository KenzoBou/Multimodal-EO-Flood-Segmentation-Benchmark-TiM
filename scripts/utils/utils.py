import numpy as np
import torch
import rasterio as rs 
import os
import sys

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
    
    image_tensor = image_tensor.squeeze(0)
    red = image_tensor[red_idx]
    green = image_tensor[green_idx]
    blue = image_tensor[blue_idx]
    stacked = torch.stack((red, green, blue), dim=0)
    normalized_rgb = _normalize_base(stacked)
    return normalized_rgb

def normalize_rgb_path(image_path:str):
    return 

if __name__ == '__main__':
    test_tensor = torch.randn((12,224,224))
    output = normalize_rgb_tensor(test_tensor)
    print(output.shape)


