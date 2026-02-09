import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, set_seed

from models.agri_vlm import AgriPerceiverVLM
from data.datasets.agri_vlm_dataset import AgriVLM_Dataset 
from training.collate_fn import agri_collate_fn 

# -----------------------------
# Configuration
# -----------------------------
set_seed(42)
DEVICE = "cuda"
LR = 1e-4
EPOCHS = 3
BATCH_SIZE = 8
GRAD_ACCUM = 2 
REPO_ROOT = "/home/vats/agri-perceiver"
CACHE_DIR = os.path.join(REPO_ROOT, "model_cache")
DATA_PATH = "/home/vats/final_train_canonical.jsonl" 

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CRITICAL: Dataset not found at {DATA_PATH}")

    # 1. Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct",
        cache_dir=CACHE_DIR,
        trust_remote_code=False
    )
    tokenizer.add_tokens(["<image>", "<image_start>", "<image_end>"])
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Models
    print("Loading Specialist Backbones...")
    phi3 = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct",
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        cache_dir=CACHE_DIR
    )
    phi3.resize_token_embeddings(len(tokenizer))
    
    siglip = AutoModel.from_pretrained(
        "google/siglip-so400m-patch14-384", 
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR
    )
    
    model = AgriPerceiverVLM(siglip, phi3, tokenizer).to(DEVICE)
    
    # 3. Load Stage 1 Connector Weights
    print("Integrating Visual Bridge Weights...")
    st1_path = os.path.join(REPO_ROOT, "stage1_connector_weights.pt")
    st1_weights = torch.load(st1_path, map_location=DEVICE)
    model.tile_embed.load_state_dict(st1_weights["tile_embed"])
    model.projector.load_state_dict(st1_weights["projector"])
    model.perceiver.load_state_dict(st1_weights["perceiver"])

    # 4. LoRA Setup
    print("Initializing LoRA adapters...")
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj", "up_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model.phi3 = get_peft_model(model.phi3, peft_config)
    
    model.siglip.requires_grad_(False)
    model.to(torch.bfloat16)

    # 5. Dataset & Loader
    dataset = AgriVLM_Dataset(
        jsonl_path=DATA_PATH,
        tokenizer=tokenizer
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=agri_collate_fn,
        num_workers=0, # Keeps shared memory safe
        pin_memory=True
    )

    optimizer = AdamW(model.parameters(), lr=LR)

    # 6. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Specialization Epoch {epoch+1}")
        optimizer.zero_grad()
        
        for step, batch in enumerate(pbar):
            # Move everything to GPU
            pixel_values = batch["pixel_values"].to(DEVICE, dtype=torch.bfloat16)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE) 
            labels = batch["labels"].to(DEVICE)
            weights = batch["sample_weight"].to(DEVICE, dtype=torch.bfloat16)

            # --- Corrected Forward Pass & Loss Calculation in main() ---

            # 1. Forward Pass (Catch both outputs and the spliced labels)
            outputs, correct_labels = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels
            )

            # 2. Preparation for Weighted Loss
            logits = outputs.logits
            correct_labels = correct_labels.to(DEVICE)

            # Shift for Causal LM: model predicts the NEXT token
            # Logits: [Batch, Seq_Len, Vocab] -> [Batch, Seq_Len-1, Vocab]
            # Labels: [Batch, Seq_Len] -> [Batch, Seq_Len-1]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = correct_labels[..., 1:].contiguous()

            # 3. Compute Loss
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )

            # 4. Apply Weights
            # Reshape loss back to [Batch, Seq_Len-1] and mean across sequence
            loss = loss.view(correct_labels.size(0), -1).mean(dim=1)
            weighted_loss = (loss * weights).mean() / GRAD_ACCUM

            # 5. Backward
            weighted_loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=weighted_loss.item() * GRAD_ACCUM)

        # Save Checkpoint
        save_path = f"agri_perceiver_specialist_e{epoch+1}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Epoch {epoch+1} saved to {save_path}")

if __name__ == "__main__":
    main()