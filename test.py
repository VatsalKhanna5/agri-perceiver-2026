import torch
from models.perceiver.perceiver_resampler import PerceiverResampler
from transformers import AutoTokenizer, AutoModelForCausalLM

B = 2
dummy = torch.randn(B, 3645, 3072)

model = PerceiverResampler()
out = model(dummy)

print(out.shape)  # Expect [2, 128, 3072]

special_tokens = {
    "additional_special_tokens": ["<image_start>", "<image_end>"]
}
tokenizer.add_special_tokens(special_tokens)
phi3.resize_token_embeddings(len(tokenizer))
