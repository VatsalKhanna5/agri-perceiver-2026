import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from models.agri_vlm import AgriPerceiverVLM
from data.datasets.agri_vlm_dataset import AgriVLM_Dataset # Use Stage 2 Dataset
from training.collate_fn import agri_collate_fn # Use Stage 2 Collator

DEVICE = "cuda"
LR = 5e-5 # Lower learning rate for fine-tuning
EPOCHS = 3
BATCH_SIZE = 8 

def train_stage2():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
    special_tokens = ["<image>", "<image_start>", "<image_end>"]
    tokenizer.add_tokens(special_tokens)

    # 1. Load Model and Stage 1 Weights
    phi3 = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", torch_dtype=torch.bfloat16, trust_remote_code=True)
    phi3.resize_token_embeddings(len(tokenizer))
    siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-384", torch_dtype=torch.bfloat16)
    
    model = AgriPerceiverVLM(siglip, phi3, tokenizer).to(DEVICE)
    
    print("Loading Stage 1 Alignment Weights...")
    st1_weights = torch.load("stage1_connector_weights.pt", map_location=DEVICE)
    model.tile_embed.load_state_dict(st1_weights["tile_embed"])
    model.projector.load_state_dict(st1_weights["projector"])
    model.perceiver.load_state_dict(st1_weights["perceiver"])

    # 2. Attach LoRA to Phi-3
    peft_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["qkv_proj", "o_proj", "down_proj", "up_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model.phi3 = get_peft_model(model.phi3, peft_config)
    model.phi3.print_trainable_parameters()

    # 3. Dataset & Loader
    dataset = AgriVLM_Dataset(
        jsonl_path="data/datasets/final_train_canonical.jsonl",
        tokenizer=tokenizer
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=agri_collate_fn)

    optimizer = AdamW(model.parameters(), lr=LR)

    # 4. Training Loop with Weighted Loss
    model.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Stage2 Epoch {epoch+1}")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(DEVICE, dtype=torch.bfloat16)
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            sample_weights = batch["sample_weight"].to(DEVICE, dtype=torch.bfloat16)

            outputs = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
            
            # Weighted Causal LM Loss
            # outputs.loss is the mean loss. We re-calculate to apply sample weights.
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Reshape loss back to batch and apply weights
            loss = loss.view(labels.size(0), -1).mean(dim=1)
            weighted_loss = (loss * sample_weights).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=weighted_loss.item())

    torch.save(model.state_dict(), "agri_perceiver_final.pt")

if __name__ == "__main__":
    train_stage2()