import terratorch
from terratorch.registry import BACKBONE_REGISTRY, TERRATORCH_BACKBONE_REGISTRY, TERRATORCH_DECODER_REGISTRY
import torch

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from config import S2_BANDS_NAMES_TO_USE

def create_terramind_base(terramind_decoder='UNetDecoder', num_classes=2):
    model = terratorch.tasks.SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory",  # Combines a backbone with necks, the decoder, and a head
        model_args={
            # TerraMind backbone
            "backbone": "terramind_v1_base", # large version: terramind_v1_large 
            "backbone_pretrained": True,
            "backbone_modalities": ["S2L1C", "S1GRD"],
            "backbone_bands": {"S2L1C":S2_BANDS_NAMES_TO_USE},
            # "backbone_bands": {"S1GRD": ["VV"]},
            
            # Necks 
            "necks": [
                {
                    "name": "SelectIndices",
                    "indices": [2, 5, 8, 11] # indices for terramind_v1_base
                    # "indices": [5, 11, 17, 23] # indices for terramind_v1_large
                },
                {"name": "ReshapeTokensToImage",
                "remove_cls_token": False},  # TerraMind is trained without CLS token, which needs to be specified.
                {"name": "LearnedInterpolateToPyramidal"}  # Some decoders like UNet or UperNet expect hierarchical features. Therefore, we need to learn a upsampling for the intermediate embedding layers when using a ViT like TerraMind.
            ],
            
            # Decoder
            "decoder": terramind_decoder,
            "decoder_channels": [512, 256, 128, 64],
            
            # Head
            "head_dropout": 0.1,
            "num_classes": num_classes,
        }    
    )
    return model

# terramind_model = create_terramind_base()
# print(terramind_model.model.encoder)

# s2l1c_tensor = torch.rand(1,9,224,224)
# s1grd_tensor = torch.rand(1,2,224,224)

# test_tensor = {"S2L1C" : s2l1c_tensor, "S1GRD" : s1grd_tensor}
# output = model(test_tensor)

# print(output.output.shape)
# print(torch.unique(torch.argmax(torch.softmax(output.output, dim=1), dim=1)))

# import matplotlib.pyplot as plt

# plt.imshow(output.output[0][0].detach().numpy())

