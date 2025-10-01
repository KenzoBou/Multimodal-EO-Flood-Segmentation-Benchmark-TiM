import torch 
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GPTQConfig, AutoProcessor
import config
import dataset
import json
import random
from torchvision import transforms
import numpy as np
import tempfile


model_name = "Qwen/Qwen-VL-Chat-Int4"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# quantization_config = GPTQConfig(
#     bits=4, # On charge un modèle 4-bit
#     dataset=None, # Pas besoin de dataset pour l'inférence
#     tokenizer=tokenizer,
#     use_exllama=False, # Utilise le kernel de base, plus stable
# )

# --- CORRECTION : Chargement avec la bonne configuration ---
# On passe la configuration GPTQ.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True
).eval()

import tempfile

def analyze_image_clarity(rgb_tensor: torch.Tensor) -> dict:
    if rgb_tensor.is_cuda:
        rgb_tensor = rgb_tensor.cpu()
    rgb_numpy = rgb_tensor.numpy()
    rgb_numpy = np.clip(rgb_numpy, 0, 1)
    rgb_numpy = (rgb_numpy * 255).astype(np.uint8)
    rgb_numpy = np.transpose(rgb_numpy, (1, 2, 0))  # (H, W, C)
    pil_image = Image.fromarray(rgb_numpy, mode='RGB').resize((224, 224))
    # ...conversion image...
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil_image.save(tmp.name)
        image_path = tmp.name

    prompt = """
    Analyze the RGB satellite image provided. Determine the level of cloud coverage.
    Respond in JSON format with two keys:
    1. 'cloud_coverage: a string which can be 'clear', 'partly_cloudy' or 'overcast'
    2. 'reasoning': a brief sentence explaining your choice.
    """

    query = tokenizer.from_list_format([
        {'image': image_path},
        {'text': prompt},
    ])

    # Utilise la méthode chat !
    response = model.chat(tokenizer, query, history=None)
    print(response.keys)

    try:
        json_response = response[response.find('{'):response.rfind('}')+1]
        return json.loads(json_response)
    except:
        return {'cloud_coverage': 'unknown', 'reasoning': response}

val_dataset = dataset.Sen1FloodsDataset(
    root_dir=config.ROOT_DIR,
    bands=config.S2_BANDS_TO_USE,
    modalities=['S2','S1'],
    split='val'
)

random_idx = [random.randint(0, len(val_dataset)-1) for k in range(5)]
for idx in random_idx:
    image_tensor, _ = val_dataset[idx]
    print(image_tensor.shape)
    mean = torch.tensor(config.GLOBAL_MEAN_BANDS_TO_USE).view(-1,1,1)
    std = torch.tensor(config.GLOBAL_STD_BANDS_TO_USE).view(-1,1,1)
    denormalized_tensor = image_tensor*std + mean
    image_rgb = denormalized_tensor[[3,2,1],:,:]
    rgb_numpy = image_rgb.numpy()
    min_vals, max_vals = np.percentile(rgb_numpy, [2, 98], axis=(1, 2), keepdims=True)
    scaled_rgb_numpy = (rgb_numpy - min_vals) / (max_vals - min_vals)
    scaled_rgb_numpy = np.clip(scaled_rgb_numpy, 0, 1)

    image_for_vlm = torch.from_numpy(scaled_rgb_numpy).float() #we convert it back to tensor for the VLM analysis
    vlm_output = analyze_image_clarity(image_for_vlm)
    print(vlm_output)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from PIL import Image
import tempfile

model_name = "Qwen/Qwen-VL-Chat-Int4"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).eval()

# Crée une image RGB 224x224 pour le test
pil_image = Image.new("RGB", (224, 224), color=(255, 255, 255))
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    pil_image.save(tmp.name)
    image_path = tmp.name

prompt = "Describe the image."

query = tokenizer.from_list_format([
    {"image": image_path},
    {"text": prompt}
])

response = model.chat(tokenizer, query, history=None)
print(response)