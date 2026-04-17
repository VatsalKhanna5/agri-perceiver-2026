"""
Specialist dataset for Stage 2 fine-tuning.

Image–JSON report pairs with sample weights for structured pathology output.
"""

import json
import os

import cv2
import torch
from torch.utils.data import Dataset


class SpecialistDataset(Dataset):
    """
    Stage 2 specialist dataset: image + structured JSON report.

    Args:
        jsonl_path: Path to JSONL with {image, canonical_report, sample_weight} entries.
        tokenizer: HuggingFace tokenizer.
        image_size: Target tile resolution.
        data_root: Root directory for images.
        max_length: Maximum token sequence length.
    """

    def __init__(self, jsonl_path: str, tokenizer, image_size: int = 384,
                 data_root: str = ".", max_length: int = 2048):
        with open(jsonl_path) as f:
            self.samples = [json.loads(line) for line in f]

        self.tokenizer = tokenizer
        self.image_size = image_size
        self.data_root = data_root
        self.max_length = max_length

    def tile_image(self, img):
        h, w = img.shape[:2]
        tiles = [
            img[: h // 2, : w // 2],
            img[: h // 2, w // 2 :],
            img[h // 2 :, : w // 2],
            img[h // 2 :, w // 2 :],
            cv2.resize(img, (w, h)),
        ]
        processed = []
        for t in tiles:
            t = cv2.resize(t, (self.image_size, self.image_size))
            t = torch.from_numpy(t).permute(2, 0, 1).float() / 255.0
            processed.append(t)
        return torch.stack(processed)

    @staticmethod
    def build_prompt() -> str:
        return (
            "<|user|>\n<image>\nYou are an agricultural pathology AI. "
            "Analyze this leaf and provide a structured lab report in JSON.\n<|assistant|>\n"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_filename = os.path.basename(sample["image"])
        img_path = os.path.join(self.data_root, img_filename)

        img = cv2.imread(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self.samples))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixel_values = self.tile_image(img)

        prompt = self.build_prompt()
        target_json = json.dumps(sample["canonical_report"])
        full_text = prompt + target_json + self.tokenizer.eos_token

        tokenized = self.tokenizer(
            full_text, return_tensors="pt", padding=False,
            truncation=True, max_length=self.max_length,
        )
        input_ids = tokenized.input_ids[0]
        labels = input_ids.clone()

        prompt_len = self.tokenizer(prompt, return_tensors="pt").input_ids.size(1)
        labels[:prompt_len] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "sample_weight": torch.tensor(sample.get("sample_weight", 1.0), dtype=torch.float32),
        }
