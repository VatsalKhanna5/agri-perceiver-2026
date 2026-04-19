#!/usr/bin/env python3
"""Download baseline model weights into HF cache."""
import os
import sys

os.environ["HF_HOME"] = os.path.expanduser("~/hf_cache")

from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel

MODELS = {
    "llava_next": {
        "name": "llava-hf/llava-v1.6-mistral-7b-hf",
        "cls": "auto_causal",
    },
    "internvl2": {
        "name": "OpenGVLab/InternVL2-8B",
        "cls": "auto_model",
    },
    # Gemma-3-12B-IT is large (~24GB) — download last
    "gemma3": {
        "name": "google/gemma-3-12b-it",
        "cls": "auto_causal",
    },
}

def main():
    which = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())

    for key in which:
        if key not in MODELS:
            print(f"Unknown model: {key}. Choose from {list(MODELS.keys())}")
            continue

        cfg = MODELS[key]
        name = cfg["name"]
        print(f"\n=== Downloading {key}: {name} ===")

        try:
            # Always download tokenizer/processor
            print(f"  Downloading tokenizer...")
            if key == "internvl2":
                AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            elif key == "llava_next":
                from transformers import LlavaNextProcessor
                LlavaNextProcessor.from_pretrained(name)
            else:
                AutoProcessor.from_pretrained(name)

            # Download model weights
            print(f"  Downloading model weights (this may take a while)...")
            if cfg["cls"] == "auto_model":
                AutoModel.from_pretrained(name, trust_remote_code=True)
            else:
                AutoModelForCausalLM.from_pretrained(name)

            print(f"  [OK] {key} downloaded successfully")

        except Exception as e:
            print(f"  [FAIL] {key}: {e}")

    print("\n=== All downloads complete ===")


if __name__ == "__main__":
    main()
