import config
import dataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torch
from torchmetrics.classification import MulticlassJaccardIndex
from torch import nn

device = config.DEVICE

U_NET_MODEL = smp.Unet(
    encoder_name='resnet34',
    in_channels=config.INPUT_DIM,
    classes=config.NUM_CLASSES
).to(device)

test_dataset = dataset.Sen1FloodsDataset(
    root_dir= config.ROOT_DIR,
    bands = config.S2_BANDS_TO_USE,
    modalities = config.MODALITIES,
    split='test'
)


U_NET_MODEL.load_state_dict(torch.load(config.MODEL_PATH/'unet_resnet34.pth', map_location=config.DEVICE))


criterion = nn.CrossEntropyLoss(ignore_index=255)  

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

def test_model(model, 
               test_loader, 
               criterion,
               save_predictions=False):
    iou_metric = MulticlassJaccardIndex(num_classes=config.NUM_CLASSES, 
                                    ignore_index=255,
                                    average=None).to(config.DEVICE)

    device = config.DEVICE
    model.to(device)
    # Validation phase
    model.eval()
    iou_metric.reset()
    full_outputs = []
    full_masks = []
    with torch.no_grad():
        val_loss = 0.0
        for batch in test_loader:
            image, mask = batch
            image = image.to(device)
            mask = mask.to(device)
            mask = mask.squeeze(1)
            outputs = model(image)
            loss = criterion(outputs, mask)
            val_loss += loss.item()
            iou_metric.update(outputs, mask)
            if save_predictions:
                full_outputs.append(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu()) 
                full_masks.append(mask.cpu())
    running_iou = iou_metric.compute()
    print(f"Mask shape: {mask.shape}, Outputs shape: {outputs.shape}")
    iou_water = running_iou[1].item() if len(running_iou) > 1 else running_iou.item()
    print("IoU for water class: {:.4f}".format(iou_water))
    print("IoU for non-water class: {:.4f}".format(running_iou[0].item() if len(running_iou) > 0 else 0.0))
    if save_predictions:
        print("Saving predictions...")
        return(val_loss/len(test_loader), running_iou, (full_outputs, full_masks))
    else : 
        return(val_loss/len(test_loader), running_iou)

if __name__ == '__main__':
    print("Starting evaluation...")
    output = test_model(
    model= U_NET_MODEL,
    test_loader= test_dataloader,
    criterion=criterion,
    save_predictions=True
)
