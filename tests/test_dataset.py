from data.datasets.agri_vlm_dataset import AgriVLM_Dataset
from training.collate_fn import agri_collate_fn
from torch.utils.data import DataLoader

dataset = AgriVLM_Dataset("microsoft/Phi-3-mini-128k-instruct")
loader = DataLoader(dataset, batch_size=2, collate_fn=agri_collate_fn)

batch = next(iter(loader))

print("Pixel:", batch["pixel_values"].shape)
print("Input IDs:", batch["input_ids"].shape)
print("Labels:", batch["labels"].shape)
print("Weights:", batch["sample_weight"])
