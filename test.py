import torch
import json
import random
import cv2
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import LoraConfig, get_peft_model
from models.agri_vlm import AgriPerceiverVLM

DEVICE = "cuda"
MODEL_PATH = "microsoft/Phi-3-mini-128k-instruct"
CHECKPOINT = "agri_perceiver_specialist_e3.pt"
DATA_PATH = "/home/vats/final_train_canonical.jsonl"
IMAGE_ROOT = "/home/vats/canonical_dataset/processed_images"

def load_and_tile_image(img_path, size=384):
    img = cv2.imread(img_path)
    if img is None: raise FileNotFoundError(f"Missing: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    mid_h, mid_w = h // 2, w // 2
    
    tiles = [
        img[:mid_h, :mid_w], img[:mid_h, mid_w:],
        img[mid_h:, :mid_w], img[mid_h:, mid_w:],
        cv2.resize(img, (w, h))
    ]
    
    processed = [torch.from_numpy(cv2.resize(t, (size, size))).permute(2,0,1).float()/255.0 for t in tiles]
    return torch.stack(processed).unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)

def load_specialist():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.add_tokens(["<image>", "<image_start>", "<image_end>"])
    
    phi3 = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True).to(DEVICE)
    phi3.resize_token_embeddings(len(tokenizer))
    
    siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-384", torch_dtype=torch.bfloat16).to(DEVICE)
    model = AgriPerceiverVLM(siglip, phi3, tokenizer).to(DEVICE)

    # Re-apply LoRA
    peft_config = LoraConfig(
        r=32, lora_alpha=64,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj", "up_proj"],
        lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
    )
    model.phi3 = get_peft_model(model.phi3, peft_config)
    
    print(f"Loading weights from {CHECKPOINT}...")
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    return model, tokenizer

@torch.no_grad()
def run_test(num=5):
    model, tokenizer = load_specialist()
    with open(DATA_PATH, 'r') as f:
        samples = [json.loads(line) for line in f]
    
    for i, sample in enumerate(random.sample(samples, num)):
        print(f"\n--- TEST SAMPLE {i+1} ---")
        img_path = os.path.join(IMAGE_ROOT, os.path.basename(sample["image"]))
        
        pixel_values = load_and_tile_image(img_path)
        prompt = "<|user|>\n<image>\nYou are an agricultural pathology AI. Analyze this leaf and provide a structured lab report in JSON.\n<|assistant|>\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        visual_latents = model.encode_images(pixel_values)
        
        input_ids = inputs.input_ids
        curr_mask = inputs.attention_mask.to(torch.long)
        
        generated_text = ""
        for _ in range(256):
            outputs, _ = model.splice_and_forward(input_ids, curr_mask, visual_latents)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            
            if next_token.item() == tokenizer.eos_token_id: break
            
            word = tokenizer.decode(next_token[0])
            generated_text += word
            print(word, end="", flush=True)
            
            input_ids = torch.cat([input_ids, next_token.to(DEVICE)], dim=1)
            curr_mask = torch.cat([curr_mask, torch.ones((1, 1), device=DEVICE, dtype=torch.long)], dim=1)

        print(f"\n\nGROUND TRUTH:\n{json.dumps(sample['canonical_report'], indent=2)}")

if __name__ == "__main__":
    run_test(5)