import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


# -------------------------------------------------
# BASE PHI-3 (Stage 1: Frozen)
# -------------------------------------------------
def load_phi3_base(model_name="microsoft/Phi-3-mini-128k-instruct"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None  # handled by accelerate
    )
    return model


# -------------------------------------------------
# PHI-3 WITH LoRA (Stage 2)
# -------------------------------------------------
def load_phi3_lora(model_name="microsoft/Phi-3-mini-128k-instruct"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None
    )

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model
