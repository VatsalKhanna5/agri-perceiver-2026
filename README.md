## Environment Setup

```bash
python3 -m venv agri_env
source agri_env/bin/activate
```

```bash
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install transformers accelerate peft bitsandbytes timm einops datasets
pip install wandb fastapi uvicorn outlines pydantic pillow opencv-python tqdm
```

## Multi-GPU config

```bash
accelerate config
accelerate test
```
## Github Access
```bash
ssh-keygen -t ed25519 -C "agri-perceiver-remote"
cat ~/.ssh/id_ed25519.pub
```

## Run Logger
```bash
nohup python monitoring/gpu_logger.py &
```

Run logger:

```bash
nohup python monitoring/gpu_logger.py &
```

---

## Dataset Usage

Expected dataset format:

```
dataset_root/
 ├── train.jsonl
 ├── val.jsonl
 └── tiles/
     ├── img_001.pt
```

Load in training:

```python
from data.dataset_loader import AgriDataset

dataset = AgriDataset(
    jsonl_path="/workspace/exported_dataset/train.jsonl",
    tiles_dir="/workspace/exported_dataset/tiles"
)
```


