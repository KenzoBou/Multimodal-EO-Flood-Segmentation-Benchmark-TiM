import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, GPTQConfig
from PIL import Image
import tempfile
import os

model_name = "Qwen/Qwen-VL-Chat-Int4"

# 1. Spécifie la configuration de quantification si nécessaire (souvent auto-détecté mais bon pour la clarté)
# use_exllama=False est souvent plus stable
quantization_config = GPTQConfig(bits=4, use_exllama=False)

print(f"Loading tokenizer from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print(f"Loading processor from {model_name}...")
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

print(f"Loading model {model_name} with torch_dtype=torch.float16 and device_map='auto'...")
# 2. Ajoute torch_dtype et device_map="auto"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16, # Utilise bfloat16 si ton GPU (Ampere+) le supporte
    # device_map="auto", # C'est crucial pour la gestion de la mémoire et des périphériques
    quantization_config=quantization_config # Passer explicitement la config de quantification
).eval()
print("Model loaded successfully.")

# Crée une image RGB 224x224 pour le test
pil_image = Image.new("RGB", (224, 224), color=(255, 255, 255))



image_path = None
try:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil_image.save(tmp.name)
        image_path = tmp.name
    print(f"Image saved to temporary file: {image_path}")

    prompt = "Describe the image."
    print(f"Preparing query with prompt: '{prompt}' and image from {image_path}")

    query = tokenizer.from_list_format([
        {"image": image_path},
        {"text": prompt}
    ])
    

    print("Sending query to model for chat generation...")
    # response = model.chat(tokenizer, query, history=None)
    inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(model.device)
    print(inputs)
    response = model.generate(**inputs)
    print("\n--- Model Response ---")
    print(response)

except Exception as e:
    print(f"An error occurred during execution: {e}")
finally:
    if image_path and os.path.exists(image_path):
        os.remove(image_path)
        print(f"Cleaned up temporary file: {image_path}")