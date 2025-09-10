import lightning.pytorch as pl
from pathlib import Path
from torchvision import transforms
import albumentations

from terratorch.datamodules import GenericMultiModalDataModule

import config
from utils.parser import get_args # On utilisera une version simplifiée
from lightning_wrappers import CustomSegmentationTask


def run():
    # We hardcode to test 
    BATCH_SIZE = 8 
    LEARNING_RATE = 1e-4
    INPUT_DIM = 11
    NUM_CLASSES = 2
    MAX_EPOCHS = 10
    ARCHITECTURE = 'unet'
    BACKBONE = 'resnet34'
    s2_band_indices = [config.ALL_S2_BANDS_NAMES.index(band) for band in config.S2_BANDS_NAMES_TO_USE]
    filtered_s2_means = [config.GLOBAL_MEAN_BANDS[i] for i in s2_band_indices]
    filtered_s2_stds = [config.GLOBAL_STD_BANDS[i] for i in s2_band_indices]


    data_path = config.PROJECT_ROOT / 'Sen1Floods11-Benchmark' / 'dataset' /'sen1floods11_v1.1'/ 'sen1floods11_v1.1'/ 'data'
    split_path = config.PROJECT_ROOT / 'Sen1Floods11-Benchmark' / 'dataset' /'sen1floods11_v1.1'/ 'sen1floods11_v1.1'/'splits'

    # normalize_transform = {'S2':transforms.Compose([transforms.Normalize(
    #     mean=config.GLOBAL_MEAN_BANDS, 
    #     std=config.GLOBAL_STD_BANDS
    # )]), 'S1':transforms.Compose([transforms.Normalize(
    #     mean=config.GLOBAL_MEAN_BANDS, 
    #     std=config.GLOBAL_STD_BANDS
    # )])}



    datamodule = GenericMultiModalDataModule(
        task='segmentation',
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES,
        modalities=['S2','S1'],
        num_workers=11,

        dataset_bands={
            'S2': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10','B11', 'B12'],
            'S1': ['VV', 'VH']
        },
        output_bands={
            'S2': config.S2_BANDS_NAMES_TO_USE,
            'S1': ['VV', 'VH']
        },
        # concat_bands=True,
        means={
            'S2':filtered_s2_means,
            'S1':config.GLOBAL_MEAN_BANDS[-2:]
        },
        stds={
            'S2':filtered_s2_stds,
            'S1':config.GLOBAL_STD_BANDS[-2:]
        },
        train_transform=None,
        # train_transform=[
        #     albumentations.D4(), # Random flips and rotation
        #     albumentations.pytorch.transforms.ToTensorV2(),
        # ],
        val_transform=None,  # Using ToTensorV2() by default if not provided
        test_transform=None,

        #Roots to train and val datasets (they are the same before text split)
        train_data_root={
            'S2':data_path/'S2L1CHand',
            'S1':data_path/'S1GRDHand'
        },
        train_label_data_root=data_path/'LabelHand',
        val_data_root={
            'S2':data_path/'S2L1CHand',
            'S1':data_path/'S1GRDHand'
        },
        val_label_data_root=data_path/'LabelHand',
        
        test_data_root={
            'S2':data_path/'S2L1CHand',
            'S1':data_path/'S1GRDHand'
        },
        test_label_data_root=data_path/'LabelHand',
        train_split=split_path/'flood_train_data.txt',
        val_split=split_path/'flood_valid_data.txt',
        test_split=split_path/'flood_test_data.txt',

         
        no_label_replace=255, # Replace -1 by 255
        no_data_replace=0 # Replace NaN by 0
    )

    pl_trainer = pl.Trainer(
        accelerator='auto',
        strategy='auto',
        max_epochs=MAX_EPOCHS,
        precision='16-mixed', #accelerates compute
        logger=pl.loggers.TensorBoardLogger("logs/", name=f"{ARCHITECTURE}-{BACKBONE}"),
    )

    
    model = CustomSegmentationTask(
        architecture=ARCHITECTURE,
        backbone=BACKBONE,
        learning_rate=LEARNING_RATE,
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES
    )
    
    print("Starting training...")
    pl_trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    run()