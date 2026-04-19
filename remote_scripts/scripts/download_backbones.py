"""Download SigLIP and Phi-3 backbone models to local HF cache."""
import os
os.environ["HF_HOME"] = os.path.expanduser("~/hf_cache")

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import torch

print("=== Downloading SigLIP-SO400M-patch14-384 ===")
siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-384", torch_dtype=torch.bfloat16)
print(f"SigLIP loaded: {sum(p.numel() for p in siglip.parameters())/1e6:.1f}M params")
del siglip

print("=== Downloading Phi-3-mini-128k-instruct ===")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
phi3 = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=False,
    attn_implementation="eager",
)
print(f"Phi-3 loaded: {sum(p.numel() for p in phi3.parameters())/1e6:.1f}M params")
print(f"Tokenizer vocab: {len(tokenizer)}")
del phi3, tokenizer

print("=== All backbones cached ===")
