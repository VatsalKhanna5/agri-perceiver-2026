"""
Alignment dataset for Stage 1 pretraining.

Image–caption pairs for training the perception bridge (projector + perceiver).
"""

import json
import os

import cv2
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class AlignmentDataset(Dataset):
    """
    Stage 1 alignment dataset: image + caption pairs.

    Args:
        jsonl_path: Path to JSONL with {image, caption} entries.
        tokenizer: HuggingFace tokenizer (or name to load).
        image_size: Target tile resolution.
        data_root: Root directory for resolving image paths.
    """

    def __init__(self, jsonl_path: str, tokenizer=None, image_size: int = 384, data_root: str = "."):
        self.data_root = data_root
        self.image_size = image_size

        with open(jsonl_path) as f:
            self.samples = [json.loads(line) for line in f]

        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tile_image(self, img):
        h, w = img.shape[:2]
        mid_h, mid_w = h // 2, w // 2
        tiles = [
            img[:mid_h, :mid_w],
            img[:mid_h, mid_w:],
            img[mid_h:, :mid_w],
            img[mid_h:, mid_w:],
            cv2.resize(img, (w, h)),
        ]
        processed = []
        for t in tiles:
            t = cv2.resize(t, (self.image_size, self.image_size))
            t = torch.from_numpy(t).permute(2, 0, 1).float() / 255.0
            processed.append(t)
        return torch.stack(processed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.data_root, sample["image"])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixel_values = self.tile_image(img)

        prompt = "<|user|>\n<image>\nDescribe this agricultural sample.<|assistant|>\n"
        caption = sample["caption"] + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids[0]
        caption_ids = self.tokenizer(caption, add_special_tokens=False, return_tensors="pt").input_ids[0]

        input_ids = torch.cat([prompt_ids, caption_ids])
        labels = input_ids.clone()
        labels[: len(prompt_ids)] = -100
        attention_mask = torch.ones_like(input_ids)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
