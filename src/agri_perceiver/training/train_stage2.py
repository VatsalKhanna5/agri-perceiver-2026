"""
Stage 2 — Specialist fine-tuning with LoRA.

Trains the perception bridge + LoRA adapters on the language model
using structured JSON diagnostic reports with weighted cross-entropy loss.

CLI:
    python -m agri_perceiver.training.train_stage2 \
        --data final_train_canonical.jsonl \
        --data-root /path/to/processed_images \
        --bridge-weights checkpoints/stage1_connector_weights.pt \
        --output checkpoints/specialist_e{epoch}.pt
"""

import argparse
import os

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, set_seed

from agri_perceiver.data.collate import specialist_collate_fn
from agri_perceiver.data.specialist_dataset import SpecialistDataset
from agri_perceiver.model.agri_vlm import AgriPerceiverVLM

SPECIAL_TOKENS = ["<image>", "<image_start>", "<image_end>"]


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Specialist Fine-tuning")
    parser.add_argument("--data", type=str, required=True, help="Path to specialist JSONL")
    parser.add_argument("--data-root", type=str, default=".", help="Image root directory")
    parser.add_argument("--bridge-weights", type=str, required=True, help="Stage 1 bridge weights")
    parser.add_argument("--output", type=str, default="specialist_e{epoch}.pt")
    parser.add_argument("--llm", type=str, default="microsoft/Phi-3-mini-128k-instruct")
    parser.add_argument("--vision", type=str, default="google/siglip-so400m-patch14-384")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm, trust_remote_code=False)
    tokenizer.add_tokens(SPECIAL_TOKENS)
    tokenizer.pad_token = tokenizer.eos_token

    # Backbones
    phi3 = AutoModelForCausalLM.from_pretrained(
        args.llm, torch_dtype=torch.bfloat16, trust_remote_code=False,
    )
    phi3.resize_token_embeddings(len(tokenizer))
    siglip = AutoModel.from_pretrained(args.vision, torch_dtype=torch.bfloat16)

    # Model + bridge weights
    model = AgriPerceiverVLM(siglip, phi3, tokenizer).to(device)
    bridge_weights = torch.load(args.bridge_weights, map_location=device, weights_only=True)
    model.tile_embed.load_state_dict(bridge_weights["tile_embed"])
    model.projector.load_state_dict(bridge_weights["projector"])
    model.perceiver.load_state_dict(bridge_weights["perceiver"])

    # LoRA
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj", "up_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.phi3 = get_peft_model(model.phi3, peft_config)
    model.siglip.requires_grad_(False)
    model.to(torch.bfloat16)

    # Dataset
    dataset = SpecialistDataset(
        jsonl_path=args.data, tokenizer=tokenizer, data_root=args.data_root,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=specialist_collate_fn, num_workers=0, pin_memory=True,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Train
    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Stage2 Epoch {epoch + 1}/{args.epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["sample_weight"].to(device, dtype=torch.bfloat16)

            outputs, correct_labels = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )

            logits = outputs.logits
            correct_labels = correct_labels.to(device)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = correct_labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            loss = loss.view(correct_labels.size(0), -1).mean(dim=1)
            weighted_loss = (loss * weights).mean() / args.grad_accum

            weighted_loss.backward()

            if (step + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{weighted_loss.item() * args.grad_accum:.4f}")

        save_path = args.output.format(epoch=epoch + 1)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Epoch {epoch + 1} saved to {save_path}")


if __name__ == "__main__":
    main()
