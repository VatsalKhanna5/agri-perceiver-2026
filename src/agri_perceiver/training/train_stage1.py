"""
Stage 1 — Alignment pretraining.

Trains only the perception bridge (TileEmbeddings + VisionProjector + PerceiverResampler)
while keeping both backbones frozen. Captioning loss on image-caption pairs.

CLI:
    python -m agri_perceiver.training.train_stage1 \
        --data data/alignment/generic_alignment.jsonl \
        --data-root /path/to/images \
        --output checkpoints/stage1_connector_weights.pt
"""

import argparse
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from agri_perceiver.data.alignment_dataset import AlignmentDataset
from agri_perceiver.data.collate import alignment_collate_fn
from agri_perceiver.model.agri_vlm import AgriPerceiverVLM

SPECIAL_TOKENS = ["<image>", "<image_start>", "<image_end>"]


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Alignment Training")
    parser.add_argument("--data", type=str, required=True, help="Path to alignment JSONL")
    parser.add_argument("--data-root", type=str, default=".", help="Image root directory")
    parser.add_argument("--output", type=str, default="stage1_connector_weights.pt")
    parser.add_argument("--llm", type=str, default="microsoft/Phi-3-mini-128k-instruct")
    parser.add_argument("--vision", type=str, default="google/siglip-so400m-patch14-384")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    tokenizer.add_tokens(SPECIAL_TOKENS)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    dataset = AlignmentDataset(
        jsonl_path=args.data,
        tokenizer=tokenizer,
        data_root=args.data_root,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=alignment_collate_fn, num_workers=0, pin_memory=True,
    )

    # Backbones
    phi3 = AutoModelForCausalLM.from_pretrained(
        args.llm, torch_dtype=torch.bfloat16, trust_remote_code=False,
    )
    phi3.resize_token_embeddings(len(tokenizer))
    siglip = AutoModel.from_pretrained(args.vision, torch_dtype=torch.bfloat16)

    # Model
    model = AgriPerceiverVLM(siglip, phi3, tokenizer).to(device)
    model.tile_embed.to(torch.bfloat16)
    model.projector.to(torch.bfloat16)
    model.perceiver.to(torch.bfloat16)

    # Freeze backbones
    model.siglip.requires_grad_(False)
    model.phi3.requires_grad_(False)

    optimizer = AdamW(
        list(model.tile_embed.parameters())
        + list(model.projector.parameters())
        + list(model.perceiver.parameters()),
        lr=args.lr,
    )

    # Train
    model.train()
    step_count = 0

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Stage1 Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            step_count += 1
            if args.max_steps and step_count >= args.max_steps:
                print(f"\nReached max_steps ({args.max_steps}).")
                break
        else:
            continue
        break

    # Save bridge weights only
    bridge_state = {
        "tile_embed": model.tile_embed.state_dict(),
        "projector": model.projector.state_dict(),
        "perceiver": model.perceiver.state_dict(),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(bridge_state, args.output)
    print(f"Stage 1 complete. Bridge weights saved to {args.output}")


if __name__ == "__main__":
    main()
