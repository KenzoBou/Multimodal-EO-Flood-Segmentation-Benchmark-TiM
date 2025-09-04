from terratorch.registry import BACKBONE_REGISTRY, TERRATORCH_BACKBONE_REGISTRY, TERRATORCH_DECODER_REGISTRY
import torch

terramind_model = BACKBONE_REGISTRY.build(
    'terramind_v1_base', 
    pretrained=True, 
    modalities=['S2L2A', 'S1GRD']    
)

for param in terramind_model.parameters():
    param.requires_grad = False

test_tensor = torch.rand((1,12,224,224))
output = terramind_model((test_tensor))
print(output[0].shape)

