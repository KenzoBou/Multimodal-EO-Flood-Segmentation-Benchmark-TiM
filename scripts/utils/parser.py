import argparse
from email import parser
import config

def get_args():
    parser = argparse.ArgumentParser(description='Train a segmentation model for flood detection based on S1 and S2 data')
    
    #Model parameters
    parser.add_argument('--architecture', '-a', type=str, default='unet', help='Model architecture', required=True, choices=['unet', 'deeplab_v3_plus','terramind', 'terramind_tim'])
    parser.add_argument('--backbone',  type=str, default='resnet34', help='Segmentation backbone', choices=['resnet34', 'resnet101', 'efficientnet-b0'])
    
    #Learning parameters
    parser.add_argument('--epochs', '-e', type=int, default=config.NUM_EPOCHS, help='Number of training epochs')
    parser.add_argument('--learning_strategy', '-ls', type=str, default='end_to_end', help='Determines if we do an end-to-end training or a fine-tuning process (only available for U-Net and DeeplabV3Plus)', choices=['end_to_end', 'fixed_fine_tuning', 'patience_fine_tuning'])
    parser.add_argument('--batch-size', '-b', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=config.LEARNING_RATE, help='Learning rate')
    
    #Fine tuning parameters, only enabled if  learning strategy involves fine tuning.
    parser.add_argument('--patience', '-p', type=int, default=config.FT_PATIENCE, help='Patience for fine training. Ensure that patience < epochs')
    parser.add_argument('--ratio', '-r', type=float, default=config.FT_RATIO, help='Ratio for fine training. Ensure that ratio is <1')

    return parser.parse_args()