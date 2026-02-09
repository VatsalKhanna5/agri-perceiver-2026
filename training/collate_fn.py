import torch
from torch.nn.utils.rnn import pad_sequence

def agri_collate_fn(batch):
    # 1. Handle Images
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    
    # 2. Handle Token IDs
    input_ids_list = [b["input_ids"] for b in batch]
    input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=0 # Standard for Phi-3
    )

    # 3. Create Attention Mask (CRITICAL: Fixes the KeyError)
    # 1 for actual tokens, 0 for padding
    attention_mask = (input_ids != 0).long()

    # 4. Handle Labels
    labels_list = [b["labels"] for b in batch]
    labels = pad_sequence(
        labels_list,
        batch_first=True,
        padding_value=-100 # Standard: cross-entropy ignores -100
    )

    # 5. Handle Sample Weights
    sample_weight = torch.stack([b["sample_weight"] for b in batch])

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "sample_weight": sample_weight
    }

# Keeping your alignment_collate_fn for reference
def alignment_collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100)

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }