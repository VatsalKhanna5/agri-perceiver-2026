"""Test that the model loads correctly on GPU and run a single dummy inference."""
import os, sys, time, json
os.environ["HF_HOME"] = os.path.expanduser("~/hf_cache")

import torch
import numpy as np

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

from agri_perceiver.inference.predictor import AgriPredictor, tile_image
from agri_perceiver.inference.schema import DiagnosticReport

CKPT = os.path.expanduser("~/agri-perceiver/checkpoints/specialist_e3.pt")
if not os.path.exists(CKPT):
    print(f"ERROR: Checkpoint not found at {CKPT}")
    sys.exit(1)

print(f"\n=== Loading model from {CKPT} ===")
t0 = time.time()
predictor = AgriPredictor(checkpoint_path=CKPT, device="cuda")
load_time = time.time() - t0
print(f"Model loaded in {load_time:.1f}s")

if torch.cuda.is_available():
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"GPU memory: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved")

# Dummy inference with a synthetic image
print("\n=== Running dummy inference ===")
dummy_img = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
pixel_values = tile_image(dummy_img).unsqueeze(0).to(predictor.device, dtype=torch.bfloat16)

t0 = time.time()
with torch.no_grad():
    visual_latents = predictor.model.encode_images(pixel_values)
encode_time = time.time() - t0
print(f"Image encoding: {encode_time*1000:.1f}ms")
print(f"Visual latents shape: {visual_latents.shape}")

# Quick generation test (just 10 tokens)
from agri_perceiver.inference.predictor import SPECIALIST_PROMPT
inputs = predictor.tokenizer(SPECIALIST_PROMPT, return_tensors="pt").to(predictor.device)
input_ids = inputs.input_ids
curr_mask = inputs.attention_mask.to(torch.long)

t0 = time.time()
generated = []
with torch.no_grad():
    for i in range(10):
        outputs, _ = predictor.model.splice_and_forward(input_ids, curr_mask, visual_latents)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
        generated.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.to(predictor.device)], dim=1)
        curr_mask = torch.cat([curr_mask, torch.ones((1,1), device=predictor.device, dtype=torch.long)], dim=1)
gen_time = time.time() - t0
print(f"Generated 10 tokens in {gen_time*1000:.1f}ms ({gen_time/10*1000:.1f}ms/token)")
print(f"Tokens: {predictor.tokenizer.decode(generated, skip_special_tokens=True)}")

print(f"\n=== GPU memory after inference ===")
if torch.cuda.is_available():
    print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"  Reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
    print(f"  Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

print("\n=== MODEL LOAD TEST PASSED ===")
