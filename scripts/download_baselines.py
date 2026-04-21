#!/usr/bin/env python3
"""Download baseline model weights into HF cache."""
import os
import subprocess
import sys

os.environ["HF_HOME"] = os.path.expanduser("~/hf_cache")

MODELS = {
    "llava_next": "llava-hf/llava-v1.6-mistral-7b-hf",
    "internvl2": "OpenGVLab/InternVL2-8B",
    # Qwen2-VL-7B replaces gated Gemma-3-12B-IT; non-gated, comparable scale, top-ranked VLM
    "qwen2vl": "Qwen/Qwen2-VL-7B-Instruct",
}


def _ensure_packages():
    """Install any missing deps before attempting model downloads."""
    for pkg, pip_name in [("sentencepiece", "sentencepiece"), ("timm", "timm")]:
        try:
            __import__(pkg)
        except ImportError:
            print(f"  Installing {pip_name}...")
            subprocess.run([sys.executable, "-m", "pip", "install", pip_name, "-q"], check=True)


def main():
    _ensure_packages()

    which = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())

    for key in which:
        if key not in MODELS:
            print(f"Unknown model: {key}. Choose from {list(MODELS.keys())}")
            continue

        name = MODELS[key]
        print(f"\n=== Downloading {key}: {name} ===")

        try:
            # Use snapshot_download to fetch all files without triggering optional
            # heavy dependencies (torchvision for Qwen2-VL, timm already installed above).
            print(f"  Downloading all files via snapshot_download...")
            from huggingface_hub import snapshot_download
            local_dir = snapshot_download(
                repo_id=name,
                cache_dir=os.environ["HF_HOME"],
                ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
            )
            print(f"  [OK] {key} downloaded to {local_dir}")

        except Exception as e:
            print(f"  [FAIL] {key}: {e}")

    print("\n=== All downloads complete ===")


if __name__ == "__main__":
    main()
