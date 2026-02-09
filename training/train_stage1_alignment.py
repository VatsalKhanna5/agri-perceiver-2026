import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import argparse

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from models.agri_vlm import AgriPerceiverVLM
from data.datasets.alignment_dataset import AlignmentDataset
from training.collate_fn import alignment_collate_fn

# -----------------------------
# Configuration
# -----------------------------
DEVICE = "cuda"
LR = 1e-4 
EPOCHS = 2
BATCH_SIZE = 16 
REPO_ROOT = "/home/vats/agri-perceiver"

def main():
    # Setup for the --max_steps demo argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    # -----------------------------
    # 1. Tokenizer Setup (CRITICAL: Do this first)
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
    
    # Add special tokens so they are treated as single units
    special_tokens = ["<image>", "<image_start>", "<image_end>"]
    tokenizer.add_tokens(special_tokens)
    
    # Phi-3 lacks a pad token; use EOS
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------
    # 2. Dataset Setup (Use the expanded tokenizer)
    # -----------------------------
    dataset = AlignmentDataset(
        jsonl_path=os.path.join(REPO_ROOT, "data/alignment/generic_alignment.jsonl"),
        tokenizer_name="microsoft/Phi-3-mini-128k-instruct",
        data_root="/home/vats"
    )
    # Force the dataset to use the tokenizer instance that has the new tokens
    dataset.tokenizer = tokenizer 

    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True, 
        collate_fn=alignment_collate_fn,
        num_workers=0, # Keep 0 to avoid shared memory Bus Errors
        pin_memory=True
    )

    # -----------------------------
    # 3. Model Setup
    # -----------------------------
    # Load Phi-3 in bfloat16
    phi3 = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct",
        torch_dtype=torch.bfloat16,
        trust_remote_code=False
    )
    # Resize embeddings to match the expanded tokenizer
    phi3.resize_token_embeddings(len(tokenizer))

    # Load SigLIP in bfloat16
    siglip = AutoModel.from_pretrained(
        "google/siglip-so400m-patch14-384",
        torch_dtype=torch.bfloat16
    )

    # Initialize VLM Wrapper
    model = AgriPerceiverVLM(siglip, phi3, tokenizer).to(DEVICE)

    # -----------------------------
    # 4. Precision & Freezing
    # -----------------------------
    # Ensure trainable connector stack matches bfloat16
    model.tile_embed.to(torch.bfloat16)
    model.projector.to(torch.bfloat16)
    model.perceiver.to(torch.bfloat16)

    # Freeze backbones: Only train the Perceiver/Projector/TileEmbeddings
    model.siglip.requires_grad_(False)
    model.phi3.requires_grad_(False)

    optimizer = AdamW(
        list(model.tile_embed.parameters()) +
        list(model.projector.parameters()) +
        list(model.perceiver.parameters()),
        lr=LR
    )

    # -----------------------------
    # 5. Training Loop
    # -----------------------------
    model.train()
    step_count = 0

    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Stage1 Epoch {epoch+1}")

        for batch in pbar:
            # Move batch to device
            pixel_values = batch["pixel_values"].to(DEVICE, dtype=torch.bfloat16)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # Forward pass
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

            # Handle --max_steps for demo runs
            step_count += 1
            if args.max_steps and step_count >= args.max_steps:
                print(f"\nReached max_steps ({args.max_steps}). Ending demo run.")
                return

    # -----------------------------
    # 6. Save Weights
    # -----------------------------
    trainable_state_dict = {
        "tile_embed": model.tile_embed.state_dict(),
        "projector": model.projector.state_dict(),
        "perceiver": model.perceiver.state_dict(),
    }
    torch.save(trainable_state_dict, "stage1_connector_weights.pt")
    print("Stage 1 alignment complete. Connector weights saved.")

if __name__ == "__main__":
    main()