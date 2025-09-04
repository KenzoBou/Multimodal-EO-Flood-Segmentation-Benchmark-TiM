import torch 
from tqdm import tqdm



def train_one_epoch(model, train_dataloader, device, optimizer, criterion, augmenter):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_dataloader):
        image, mask = batch
        image = image.to(device)
        mask = mask.to(device)
        mask_to_aug = mask.float()
        image_aug, mask_aug = augmenter(image, mask_to_aug)
        mask_for_loss = mask_aug.squeeze(1).long()
        optimizer.zero_grad()
        outputs = model(image_aug)
        loss = criterion(outputs, mask_for_loss)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_dataloader)

def validate_one_epoch(model, val_dataloader, device, criterion, iou_metric):
# Validation phase
    model.eval()
    iou_metric.reset()
    with torch.no_grad():
        val_loss = 0.0
        for batch in tqdm(val_dataloader):
            image, mask = batch
            image = image.to(device)
            mask = mask.to(device)
            mask = mask.squeeze(1)
            outputs = model(image)
            loss = criterion(outputs, mask)
            val_loss += loss.item()
            iou_metric.update(outputs, mask)
    running_iou = iou_metric.compute()
    val_iou_water = running_iou[1].item() if len(running_iou) > 1 else running_iou.item()
    return val_loss / len(val_dataloader), val_iou_water

