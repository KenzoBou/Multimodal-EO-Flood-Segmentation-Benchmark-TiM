import torch 
import torch.nn as nn
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
import kornia as K
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex

import config
import dataset
import train
from vision_models.unet import create_unet
from vision_models.deeplab import create_deeplab
from utils.parser import get_args

def get_best_metric(experiment_name):
    "Ask MLFlow for the best IoU"
    try: 
        runs = mlflow.search_runs(experiment_names=[experiment_name], order_by="metrics.val_iou_water DESC")
        if "metrics.val_iou_water" not in runs.columns or runs.empty or runs["metrics.val_iou_water"].isna().all():
            return -1.0
        return runs['metrics.val_iou_water'].iloc[0]
    except Exception :
        return -1.0

def main():
    device = config.DEVICE
    args = get_args()
    best_iou = get_best_metric(config.EXPERIMENT_NAME)
    print(f"Best IoU so far: {best_iou}")
    #ML Flow set up 
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    with mlflow.start_run():
        print(f"Beggining mlflow run with id {mlflow.active_run().info.run_id} and experiment id {mlflow.active_run().info.experiment_id}")
        print(vars(args).keys(), vars(args).values())
        mlflow.log_params(vars(args))

        # Model creation phase 
        if args.architecture == 'unet':
            model = create_unet(
                backbone = args.backbone,
                input_dim=config.INPUT_DIM,
                num_classes=config.NUM_CLASSES,
                model_root=None
            )

        elif args.architecture == 'deeplab_v3_plus':
                model = create_deeplab(
                backbone = args.backbone,
                input_dim=config.INPUT_DIM,
                num_classes=config.NUM_CLASSES,
                model_root=None
            )
        

        iou_metric = MulticlassJaccardIndex(num_classes=config.NUM_CLASSES, 
                                            ignore_index=255,
                                            average=None
                                            ).to(config.DEVICE)

        criterion = nn.CrossEntropyLoss(ignore_index=255)  

        augmentations = K.augmentation.AugmentationSequential(
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomVerticalFlip(p=0.5),
            K.augmentation.RandomRotation(degrees=15, p=0.5),
            data_keys=['input', 'mask']
        )

        train_dataset = dataset.Sen1FloodsDataset(
            root_dir= config.ROOT_DIR,
            bands = config.S2_BANDS_TO_USE,
            modalities = config.MODALITIES,
            split='train'
        )

        val_dataset = dataset.Sen1FloodsDataset(
            root_dir= config.ROOT_DIR,
            bands = config.S2_BANDS_TO_USE,
            modalities = config.MODALITIES,
            split='val'
        )

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )

        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )

        model.to(device)
        augmenter = augmentations.to(device)

        if args.learning_strategy == 'end_to_end':
            print("\n ---- Launching end-to-end training ---- \n")
            state = 'unfrozen'
            optimizer =  torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)        
        elif args.learning_strategy == 'fixed_fine_tuning':
            print("\n ---- Launching fixed fine-tuning training ---- \n")
            for param in model.encoder.parameters():
                param.requires_grad = False
            state = 'frozen'
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE)

        elif args.learning_strategy == 'patience_fine_tuning':
            print("\n ---- Launching patience fine-tuning training ---- \n")      
            state = 'frozen'
            patience_count = 0
            for param in model.encoder.parameters():
                param.requires_grad = False
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE)


        best_val_loss = 10e5
        for epoch in tqdm(range(args.epochs)):
            avg_train_loss = train.train_one_epoch(model=model,
                                train_dataloader=train_dataloader,
                                device=device,
                                optimizer=optimizer,
                                criterion=criterion,
                                augmenter=augmenter)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

            # Validation phase
            avg_val_loss, val_iou_water = train.validate_one_epoch(model=model,
                                                                val_dataloader=val_dataloader,
                                                                device=device,
                                                                criterion=criterion,
                                                                iou_metric=iou_metric
                                                                )

            if args.learning_strategy == 'fixed_fine_tuning' and epoch >= int(args.ratio * args.epochs) and state == 'frozen':
                print("Threshold reached, transitioning to full training")
                state = 'unfrozen'
                for param in model.encoder.parameters():
                    param.requires_grad = True
                optimizer =  torch.optim.Adam([
                {'params':model.encoder.parameters(), 'lr':config.LEARNING_RATE/10},
                {'params':model.decoder.parameters(), 'lr':config.LEARNING_RATE}])
            
            elif args.learning_strategy == 'patience_fine_tuning' and state == 'frozen':
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_count = 0
                else :
                    patience_count += 1
                if patience_count > args.patience:
                    print("Patience exceeded, transitionning to full training")
                    for param in model.encoder.parameters():
                        param.requires_grad = True
                    optimizer =  torch.optim.Adam([
                    {'params':model.encoder.parameters(), 'lr':config.LEARNING_RATE/10},
                    {'params':model.decoder.parameters(), 'lr':config.LEARNING_RATE}])
                    state = 'unfrozen'

            mlflow.log_metric("val_iou_water", val_iou_water, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

            if val_iou_water > best_iou:
                        print("Current IoU is the best so far in this training loop.")
                        best_iou = val_iou_water
                        print(f"Saving model with IoU: {best_iou}") 
                        mlflow.pytorch.log_model(model, "champion_model")
                        mlflow.log_metric("best_global_iou_water", best_iou, step=epoch)

if __name__ == '__main__':
    output = main(
    )

    