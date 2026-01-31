import torch
import torch.nn as nn
from transformers import AutoModel


class SigLIPWrapper(nn.Module):
    """
    Wraps SigLIP vision encoder to output patch tokens
    Shape: [B, 5, 729, 1152]
    """

    def __init__(self, model_name="google/siglip-so400m-patch14-384"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, images):
        """
        images: [B, 5, 3, 384, 384]
        """
        B, T, C, H, W = images.shape
        images = images.view(B * T, C, H, W)

        outputs = self.model.vision_model(images)
        patch_tokens = outputs.last_hidden_state  # [B*T, 729, 1152]

        patch_tokens = patch_tokens.view(B, T, 729, 1152)
        return patch_tokens


def load_siglip():
    model = SigLIPWrapper()
    return model
