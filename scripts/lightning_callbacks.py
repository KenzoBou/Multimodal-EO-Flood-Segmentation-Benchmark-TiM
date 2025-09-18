
from lightning.pytorch import Callback
import torch
import lightning.pytorch as pl
import matplotlib.pyplot as plt 
import numpy as np

import config

class EncoderFineTuning(Callback):
    "Callback to unfreeze the encoder when the validation loss has not decreased for a given number of epochs (patience)"
    def __init__(self, patience=2, unfreeze_lr=1e-4):
        super().__init__()
        self.patience = patience
        self.unfreeze_lr = unfreeze_lr
        self.is_frozen = True
        self.patience_counter = 0
        self.best_frozen_val_loss = float('inf')
    
    def on_fit_start(self, trainer:pl.Trainer, pl_module:pl.LightningModule):
        #We want to plot some information 
        print(f"FINE TUNING CALLBACK, BEGINNING OF THE FIT")
        self._freeze_encoder(pl_module=pl_module)
        trainer.optimizers[0].param_groups.clear() # We clear the optimizer to avoid any issue
        trainer.optimizers[0].add_param_group({
            'params':filter(lambda p: p.requires_grad, pl_module.model.parameters())
        })

    
    def on_validation_epoch_end(self, trainer:pl.Trainer, pl_module:pl.LightningModule):
        if not self.is_frozen:
            return
        
        current_val_loss = trainer.callback_metrics.get('val_loss')
        if current_val_loss == None:
            return
        
        if current_val_loss < self.best_frozen_val_loss:
            self.best_frozen_val_loss = current_val_loss
            self.patience_counter = 0 
        else : 
            self.patience_counter+=1
            print(f'Patience counter increased and is now at {self.patience_counter}')

        if self.patience_counter >= self.patience:
            print(f"FINE TUNING CALLBACK, PATIENCE {self.patience} REACHED, TRANSITIONING TO ENCODER UNFREEZING")
            self._unfreeze_and_update_optimizer(trainer, pl_module=pl_module)
            self.is_frozen=False
            self.patience_counter = 0 
    
    def _get_encoder(self, pl_module:pl.LightningModule):
        if 'terramind' in str(pl_module.hparams.architecture):
            return pl_module.model.model.encoder
        else :
            return pl_module.model.encoder

    def _freeze_encoder(self, pl_module):
        encoder = self._get_encoder(pl_module)
        for param in encoder.parameters():
            param.requires_grad = False
    
    def _unfreeze_and_update_optimizer(self, trainer, pl_module):
        encoder = self._get_encoder(pl_module)

        # 1, unfreeze the weights 
        for param in encoder.parameters():
            param.requires_grad = True 
        
        # 2, retrieve learning rate and current optimizer
        optimizer = trainer.optimizers[0]
        decoder_lr = optimizer.param_groups[0]['lr']

        # 3 update optimizer
        optimizer.add_param_group({
            'params': encoder.parameters(),
            'lr': self.unfreeze_lr
        })
        print(f"UNFREEZING ENCODER | ENCODER LR {self.unfreeze_lr}, DECODER LR {decoder_lr}")
        
class PredictionLogger(pl.Callback):
    def __init__(self, num_instances:int = 3):
        super().__init__()
        self.num_samples=num_instances

    def on_validation_epoch_end(self, trainer:pl.Trainer, pl_module:pl.LightningModule):
        if not hasattr(pl_module, 
                       'plot_validation_step'):
            print('No validation outputs found, returning None')
            return 
        
        # We retrieve the batch
        outputs = pl_module.plot_validation_step[0]
        image = outputs['images'].cpu()
        preds = outputs['preds'].cpu()
        masks = outputs['masks'].cpu()
        
        fig, axes = plt.subplots(nrows=3, ncols=self.num_samples)
        fig.suptitle(f'Epoch {trainer.current_epoch}: Predicitions vs. Ground Truth')

        for i in range(self.num_samples):
            mean = torch.tensor(config.GLOBAL_MEAN_BANDS_TO_USE).view((-1,1,1))
            std = torch.tensor(config.GLOBAL_STD_BANDS_TO_USE).view((-1,1,1))
            image_denorm = (image[i] * std + mean)
            image_rgb = image_denorm[[4,3,2],:,:].numpy()
            min_rgb, max_rgb = np.percentile(image_rgb, (2, 98), axis=(1, 2), keepdims=True)
            normalized_rgb = ((image_rgb - min_rgb) / (max_rgb - min_rgb)).transpose(1,2,0)
            axes[0,i].imshow(normalized_rgb)
            axes[0,i].set_title(f'Input Image (RGB) {i}')
            axes[0,i].axis('off')
            axes[1,i].imshow(masks[i], cmap='gray', vmin=0, vmax=1)
            axes[1,i].set_title(f'Ground Truth Mask {i}')
            axes[1,i].axis('off')
            axes[2,i].imshow(preds[i], cmap='gray', vmin=0, vmax=1)
            axes[2,i].set_title(f'Predicted Mask {i}')
            axes[2,i].axis('off')

        if isinstance(trainer.logger, pl.loggers.MLFlowLogger):
            trainer.logger.experiment.log_figure(
                run_id=trainer.logger.run_id,
                figure=fig,
                artifact_file=f'predictions_epoch_{trainer.current_epoch}.png'
            )
        plt.close(fig)


if __name__ == '__main__':
    pred_log = PredictionLogger(5)