"""
Collate functions for DataLoader batching.
"""

import torch
from torch.nn.utils.rnn import pad_sequence


def alignment_collate_fn(batch):
    """Collate for Stage 1 alignment training."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100)

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def specialist_collate_fn(batch):
    """Collate for Stage 2 specialist training with sample weights."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=0)
    attention_mask = (input_ids != 0).long()
    labels = pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100)
    sample_weight = torch.stack([b["sample_weight"] for b in batch])

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "sample_weight": sample_weight,
    }
