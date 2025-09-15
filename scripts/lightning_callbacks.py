from lightning.pytorch import Callback
import lightning.pytorch as pl

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

    def _freeze_encoder(self, pl_module):
        for param in pl_module.model.encoder.parameters():
            param.requires_grad = False
    
    def _unfreeze_and_update_optimizer(self, trainer, pl_module):
        # 1, unfreeze the weights 
        for param in pl_module.model.encoder.parameters():
            param.requires_grad = True 
        
        # 2, retrieve learning rate and current optimizer
        optimizer = trainer.optimizers[0]
        decoder_lr = optimizer.param_groups[0]['lr']

        # 3 update optimizer
        optimizer.add_param_group({
            'params':pl_module.model.encoder.parameters(),
            'lr': self.unfreeze_lr
        })
        print(f"UNFREEZING ENCODER | ENCODER LR {self.unfreeze_lr}, DECODER LR {decoder_lr}")
        
