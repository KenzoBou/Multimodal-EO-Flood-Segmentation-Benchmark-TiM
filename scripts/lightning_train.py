import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
from torchvision import transforms
from lightning.pytorch.loggers import MLFlowLogger
import albumentations

from terratorch.datamodules import GenericMultiModalDataModule

import config
from utils.parser import get_args 
from lightning_wrappers import CustomSegmentationTask
from lightning_callbacks import EncoderFineTuning, PredictionLogger


def run():
    # We hardcode to test 
    args = get_args()
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    INPUT_DIM = config.INPUT_DIM
    NUM_CLASSES = config.NUM_CLASSES
    MAX_EPOCHS = args.epochs
    ARCHITECTURE = args.architecture
    BACKBONE = args.backbone
    PATIENCE = args.patience
    TERRAMIND_DECODER = args.terramind_decoder

    concat_bands = False if 'terramind' in ARCHITECTURE else True
    print(f"------------------Concat bands is set to {concat_bands}-----------------------")


    run_name = f'{ARCHITECTURE}|{BACKBONE}|{LEARNING_RATE}'
    mlflow_log = MLFlowLogger(
        experiment_name=config.EXPERIMENT_NAME,
        run_name=run_name
    )

    checkpoint_callback = ModelCheckpoint(dirpath=config.MODEL_PATH,
                                          filename=f"{ARCHITECTURE}-{BACKBONE}"+"-{epoch:02d}-{val_iou_water:.2f}",
                                          monitor='val_iou_water',
                                          mode='max',)
    fine_tuning_callback = EncoderFineTuning(
        patience=PATIENCE,
        unfreeze_lr=LEARNING_RATE/5 #to modify later
    )


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
        num_workers=4,

        dataset_bands={
            'S2': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10','B11', 'B12'],
            'S1': ['VV', 'VH']
        },
        output_bands={
            'S2': config.S2_BANDS_NAMES_TO_USE,
            'S1': ['VV', 'VH']
        },
        concat_bands=concat_bands,
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

         
        no_label_replace=-1, # Replace -1 by 255
        no_data_replace=0 # Replace NaN by 0
    )

    pl_trainer = pl.Trainer(
        accelerator='auto',
        strategy='auto',
        max_epochs=MAX_EPOCHS,
        precision='16-mixed', #accelerates compute
        logger=mlflow_log,
        callbacks=[checkpoint_callback, 
                   fine_tuning_callback,
                   PredictionLogger(num_instances=3)],
    )

    
    model = CustomSegmentationTask(
        architecture=ARCHITECTURE,
        smp_backbone=BACKBONE,
        terramind_decoder= TERRAMIND_DECODER,
        learning_rate=LEARNING_RATE,
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES
    )
    
    print("Starting training...")
    pl_trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    run()