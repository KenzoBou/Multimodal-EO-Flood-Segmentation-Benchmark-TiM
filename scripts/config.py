import torch 
from pathlib import Path
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ROOT_DIR = PROJECT_ROOT / 'Sen1Floods11-Benchmark' / 'dataset' /'sen1floods11_v1.1'/ 'sen1floods11_v1.1'/'data'
MODEL_PATH = PROJECT_ROOT / 'Sen1Floods11-Benchmark' / 'models'
S2_BANDS_TO_USE = [2, 3, 4, 5, 6, 7, 8, 11, 12]
S2_BANDS_NAMES_TO_USE = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12']
ALL_S2_BANDS_NAMES = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10','B11', 'B12']
MODALITIES = ['S1', 'S2']
INPUT_DIM = len(S2_BANDS_TO_USE) + 2 if 'S1' in MODALITIES else len(S2_BANDS_TO_USE)
EXPERIMENT_NAME = "Flood Segmentation"

# --- Paramètres d'Entraînement ---
BATCH_SIZE = 8  # À ajuster selon la VRAM de ton GPU
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
NUM_WORKERS = 4 # Pour le DataLoader, à ajuster selon ton CPU
FT_PATIENCE = 2 # Patience for the fine_tuning training
FT_RATIO = 0.2

# --- Paramètres du Modèle ---
MODEL_ARCHITECTURE = 'Unet'
MODEL_BACKBONE = 'resnet34'
NUM_CLASSES = 2 # 0: Pas d'eau, 1: Eau


#This is the means 
GLOBAL_MEAN_BANDS = [1598.8531, 1362.6958, 1327.7782, 1169.7328, 1421.3483, 2346.7397,
        2806.1011, 2587.2292, 3037.6382,  466.7194,   51.9969, 1980.6827,
        1134.9529,  -10.5458,  -17.3930]

# we always keep the S1 data in UNET and Deeplab but not in terramind

S2_MEAN_BANDS_TO_USE = [GLOBAL_MEAN_BANDS[idx] for idx in S2_BANDS_TO_USE]
GLOBAL_MEAN_BANDS_TO_USE =  S2_MEAN_BANDS_TO_USE + GLOBAL_MEAN_BANDS[-2:]

GLOBAL_STD_BANDS = [354.9994, 404.8562, 418.5381, 517.5894, 465.1054, 636.3958, 765.9728,
        739.3975, 850.8776, 143.3036,  15.9182, 693.8527, 523.0807,   3.5716,
          4.0357]

S2_STD_BANDS_TO_USE = [GLOBAL_STD_BANDS[idx] for idx in S2_BANDS_TO_USE] 
GLOBAL_STD_BANDS_TO_USE = S2_STD_BANDS_TO_USE + GLOBAL_STD_BANDS[-2:]
