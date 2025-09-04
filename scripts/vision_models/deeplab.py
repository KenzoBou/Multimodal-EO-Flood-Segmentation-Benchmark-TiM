import torch 
import segmentation_models_pytorch as smp
import config

def create_deeplab(backbone = config.MODEL_BACKBONE, 
                input_dim=config.INPUT_DIM, 
                num_classes=config.NUM_CLASSES,
                model_root=None):
    """Args : 
        backbone : A backbone, one of the config file
        input_dim : the number of S2 bands + 2 S1 bands if used
        num_classes: 2, water or not 
        model_root: the path to the model weights (only the model name, the parent file is already included), if None, the model will be initialized randomly"""
    DEEPLABV3PLUS_MODEL = smp.DeepLabV3Plus(
    encoder_name=backbone,
    in_channels=input_dim,
    classes=num_classes)
    if model_root:
        DEEPLABV3PLUS_MODEL.load_state_dict(torch.load(config.MODEL_PATH/ model_root, map_location=config.DEVICE))
    return DEEPLABV3PLUS_MODEL

