import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer
from models.agri_vlm import AgriPerceiverVLM
from data.dataset_loader import AgriDataset
from models.vision.siglip_wrapper import load_siglip
from models.llm.phi3_lora import load_phi3_base

# Accelerator Setup
accelerator = Accelerator(mixed_precision="bf16")
device = accelerator.device

# Load Models
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

siglip = load_siglip().to(device)
phi3 = load_phi3_lora().to(device)

model = AgriPerceiverVLM(siglip, phi3, tokenizer)

# Freeze everything first
for p in model.parameters():
    p.requires_grad = False

# Unfreeze bridge modules
for p in model.tile_embed.parameters():
    p.requires_grad = True

for p in model.projector.parameters():
    p.requires_grad = True

for p in model.perceiver.parameters():
    p.requires_grad = True


#dataset 
train_dataset = AgriDataset("configs/stage1_pretrain.yaml")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

#optimizer
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=0.01
)


#Prepare for Multi-GPU

model, optimizer, train_loader = accelerator.prepare(
    model, optimizer, train_loader
)


# Training Loop

model.train()

for epoch in range(1):
    for step, batch in enumerate(train_loader):

        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask, images, labels)
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        if step % 50 == 0:
            accelerator.print(f"Step {step} | Loss {loss.item():.4f}")


accelerator.wait_for_everyone()
unwrapped = accelerator.unwrap_model(model)

torch.save({
    "perceiver": unwrapped.perceiver.state_dict(),
    "projector": unwrapped.projector.state_dict(),
    "tile_embed": unwrapped.tile_embed.state_dict(),
}, "checkpoints/stage1_alignment.pt")
