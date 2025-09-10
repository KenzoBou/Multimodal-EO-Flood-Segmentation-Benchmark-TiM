import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics.classification import MulticlassJaccardIndex

from vision_models.unet import create_unet
from vision_models.deeplab import create_deeplab

class CustomSegmentationTask(pl.LightningModule):
    def __init__(self, architecture, backbone, learning_rate, input_dim,num_classes):
        super().__init__()
        self.save_hyperparameters()
        # We'll use the factories to create the architecture of the models
        if self.hparams.architecture == 'unet':
            self.model = create_unet(
                backbone=self.hparams.backbone,
                input_dim=self.hparams.input_dim,
                num_classes=self.hparams.num_classes,
                model_root=None
            )

        elif self.hparams.architecture == 'deeplab_v3_plus':
            self.model = create_deeplab(
                backbone=self.hparams.backbone,
                input_dim=self.hparams.input_dim,
                num_classes=self.hparams.num_classes,
                model_root=None
            )
        else :
            raise ValueError(f'Architecture {self.hparams.architecture} is not supported')

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.iou_metric = MulticlassJaccardIndex(
        num_classes=self.hparams.num_classes,
        average='none', #We want per class IoU for benchmark
        ignore_index=-1
    )

    def forward(self, x):
        return(self.model(x))
    
    def _common_steps(self, batch, batch_idx):
        image, mask = batch['image'], batch['mask']
        
        mask = batch['mask'].squeeze(1)
        
        outputs = self(image)
        loss = self.loss_fn(outputs, mask)
        return loss, outputs, mask 

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._common_steps(batch, batch_idx)
        
        self.log(name='train_loss',
                 value=loss,
                 prog_bar=True,
                 on_step=True,
                 on_epoch=True)
    
        return loss # lighting does the step and backprop itself

    def validation_step(self, batch, batch_idx):
        loss, output, mask = self._common_steps(batch, batch_idx)
        self.iou_metric.update(output, mask)

        self.log(name='val_loss',
                 value=loss,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True)
    
    def on_validation_epoch_end(self):
        iou = self.iou_metric.compute()
        val_iou_water = iou[1]
        self.log(name='val_iou_water',
                 value=val_iou_water,
                 on_epoch=True,
                 prog_bar=True)
        self.iou_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

# if __name__ == '__main__':
#     task = CustomSegmentationTask(
#         architecture='unet',
#         backbone='resnet34',
#         learning_rate=1e-3,
#         input_dim=12, 
#         num_classes=2
#     )
#     image_tensor = torch.rand((4,12,224,224))
#     mask_tensor = torch.randint(0, 2,(4,1,224,224))
#     fake_batch = (image_tensor, mask_tensor)
#     loss = task.training_step(fake_batch, 0)
#     print(f"Test du training_step réussi. Loss calculée : {loss.item()}")
    
#     # Teste un validation step
#     task.validation_step(fake_batch, 0)
#     print("Test du validation_step réussi.")