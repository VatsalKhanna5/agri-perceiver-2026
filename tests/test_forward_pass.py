import torch
import torch.nn as nn
from models.agri_vlm import AgriPerceiverVLM

# -------------------------------------------------
# Dummy SigLIP encoder
# -------------------------------------------------
class DummySigLIP(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B = x.size(0)
        return torch.randn(B, 5, 729, 1152, device=x.device)


# -------------------------------------------------
# Dummy Phi-3 model
# -------------------------------------------------
class DummyPhi3(nn.Module):
    def __init__(self, vocab_size=32000, dim=3072):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds, attention_mask=None, labels=None):
        logits = self.lm_head(inputs_embeds)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        return type("Output", (), {"loss": loss, "logits": logits})


# -------------------------------------------------
# Dummy tokenizer
# -------------------------------------------------
class DummyTokenizer:
    def __init__(self):
        self.token_map = {"<image>": 1, "<image_start>": 2, "<image_end>": 3}

    def convert_tokens_to_ids(self, token):
        return self.token_map[token]


# -------------------------------------------------
# Instantiate components
# -------------------------------------------------
siglip_model = DummySigLIP()
phi3_model = DummyPhi3()
tokenizer = DummyTokenizer()

model = AgriPerceiverVLM(siglip_model, phi3_model, tokenizer)

# -------------------------------------------------
# Fake batch
# -------------------------------------------------
B = 2
images = torch.randn(B, 5, 3, 384, 384)
input_ids = torch.randint(0, 1000, (B, 128))
input_ids[:, 10] = 1  # insert <image> token
attention_mask = torch.ones_like(input_ids)
labels = torch.randint(0, 1000, (B, 128))

# -------------------------------------------------
# Forward pass
# -------------------------------------------------
out = model(input_ids, attention_mask, images, labels)

print("LOSS:", out.loss)
print("Logits shape:", out.logits.shape)
