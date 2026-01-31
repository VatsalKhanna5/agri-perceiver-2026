import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class AgriDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tiles_dir: str,
        tokenizer_name: str = "microsoft/Phi-3-mini-128k-instruct",
        max_length: int = 2048,
    ):
        self.tiles_dir = tiles_dir
        self.max_length = max_length

        # Load metadata
        self.samples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.samples)

    def _build_prompt(self, conversation):
        """
        Convert conversation format into training prompt.
        """
        human = conversation[0]["value"]
        assistant = conversation[1]["value"]

        prompt = f"<image>\n{human}\n"
        return prompt, assistant

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load tiles
        image_id = sample["id"]
        tile_path = os.path.join(self.tiles_dir, f"{image_id}.pt")
        images = torch.load(tile_path)  # [5,3,384,384]

        # Build text
        prompt, answer = self._build_prompt(sample["conversations"])

        full_text = prompt + answer

        tokenized = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = tokenized.input_ids.squeeze(0)
        attention_mask = tokenized.attention_mask.squeeze(0)

        # LOSS MASKING
        labels = input_ids.clone()
        prompt_len = len(self.tokenizer(prompt).input_ids)

        labels[:prompt_len] = -100  # mask prompt + <image> token

        return {
            "images": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
