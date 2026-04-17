"""Backbone loading utilities for vision encoders and language models."""

import torch
from transformers import AutoModel, AutoModelForCausalLM


def load_vision_encoder(model_name: str = "google/siglip-so400m-patch14-384", dtype=torch.bfloat16):
    """Load a pretrained vision encoder."""
    return AutoModel.from_pretrained(model_name, torch_dtype=dtype)


def load_language_model(model_name: str = "microsoft/Phi-3-mini-128k-instruct", dtype=torch.bfloat16):
    """Load a pretrained causal language model."""
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=False,
    )


def load_language_model_with_lora(model_name_or_path: str, lora_config=None, dtype=torch.bfloat16):
    """Load a language model and apply LoRA adapters."""
    from peft import LoraConfig, get_peft_model

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=False,
    )

    if lora_config is None:
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj", "up_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )

    return get_peft_model(model, lora_config)
