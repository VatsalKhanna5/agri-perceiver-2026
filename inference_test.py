import torch
import cv2
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from models.agri_vlm import AgriPerceiverVLM

DEVICE = "cuda"
MODEL_PATH = "microsoft/Phi-3-mini-128k-instruct"
VISION_PATH = "google/siglip-so400m-patch14-384"
WEIGHTS_PATH = "stage1_connector_weights.pt"
TEST_IMAGE = "/home/vats/canonical_dataset/processed_images/agri_000000.jpg"

def load_and_tile_image(img_path, image_size=384):
    img = cv2.imread(img_path)
    if img is None: raise FileNotFoundError(f"Image not found at {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    mid_h, mid_w = h // 2, w // 2
    
    tiles = [
        img[:mid_h, :mid_w], img[:mid_h, mid_w:],
        img[mid_h:, :mid_w], img[mid_h:, mid_w:],
        cv2.resize(img, (w, h))
    ]
    
    processed = []
    for t in tiles:
        t = cv2.resize(t, (image_size, image_size))
        t = torch.from_numpy(t).permute(2, 0, 1).float() / 255.0
        processed.append(t)
    
    # Return as bfloat16
    return torch.stack(processed).unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)

@torch.no_grad()
def run_inference():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.add_tokens(["<image>", "<image_start>", "<image_end>"])
    
    print("Loading models to H200...")
    # Use native implementation to avoid DynamicCache errors
    phi3 = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=False
    ).to(DEVICE)
    phi3.resize_token_embeddings(len(tokenizer))
    
    siglip = AutoModel.from_pretrained(VISION_PATH, torch_dtype=torch.bfloat16).to(DEVICE)
    
    model = AgriPerceiverVLM(siglip, phi3, tokenizer).to(DEVICE)

    print(f"Loading weights from {WEIGHTS_PATH}...")
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.tile_embed.load_state_dict(state_dict["tile_embed"])
    model.projector.load_state_dict(state_dict["projector"])
    model.perceiver.load_state_dict(state_dict["perceiver"])
    
    # FINAL GUARD: Force entire model to bfloat16 after weight loading
    model.to(torch.bfloat16)
    model.eval()

    pixel_values = load_and_tile_image(TEST_IMAGE)
    prompt = "<|user|>\n<image>\nDescribe the visual features of this leaf in detail.<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Pre-calculate visual features
    visual_latents = model.encode_images(pixel_values).to(torch.bfloat16)
    
    input_ids = inputs.input_ids
    curr_mask = inputs.attention_mask
    
    print("\n--- Model Output ---")
    # Manual Greedy Search Loop
    for _ in range(150):
        outputs = model.splice_and_forward(
            input_ids=input_ids,
            attention_mask=curr_mask,
            visual_latents=visual_latents,
            labels=None
        )
        
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        
        token_str = tokenizer.decode(next_token[0])
        print(token_str, end="", flush=True)

        if next_token.item() == tokenizer.eos_token_id:
            break
            
        input_ids = torch.cat([input_ids, next_token.to(DEVICE)], dim=1)
        curr_mask = torch.cat([curr_mask, torch.ones((1, 1), device=DEVICE, dtype=curr_mask.dtype)], dim=1)

    print("\n\n--- Inference Complete ---")

if __name__ == "__main__":
    run_inference()