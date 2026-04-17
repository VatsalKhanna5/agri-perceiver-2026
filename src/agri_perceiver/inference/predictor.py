"""
AgriPredictor — Single-call inference API for AgriPerceiver VLM.

Usage:
    from agri_perceiver.inference.predictor import AgriPredictor

    predictor = AgriPredictor("checkpoints/specialist_e3.pt")
    report = predictor.predict("path/to/leaf.jpg")

CLI:
    agri-predict --image leaf.jpg --checkpoint checkpoints/specialist_e3.pt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from agri_perceiver.inference.schema import DiagnosticReport
from agri_perceiver.model.agri_vlm import AgriPerceiverVLM


# Default backbone identifiers
DEFAULT_LLM = "microsoft/Phi-3-mini-128k-instruct"
DEFAULT_VISION = "google/siglip-so400m-patch14-384"
SPECIAL_TOKENS = ["<image>", "<image_start>", "<image_end>"]

SPECIALIST_PROMPT = (
    "<|user|>\n<image>\n"
    "You are an agricultural pathology AI. "
    "Analyze this leaf and provide a structured lab report in JSON.\n"
    "<|assistant|>\n"
)


def tile_image(img: np.ndarray, image_size: int = 384) -> torch.Tensor:
    """
    AnyRes tiling: split image into 4 quadrants + 1 global view.

    Args:
        img: RGB numpy array [H, W, 3].
        image_size: Target tile size (default 384 for SigLIP).

    Returns:
        [5, 3, image_size, image_size] tensor normalized to [0, 1].
    """
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
        t = cv2.resize(t, (image_size, image_size))
        t = torch.from_numpy(t).permute(2, 0, 1).float() / 255.0
        processed.append(t)

    return torch.stack(processed)


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk with robust error handling."""
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Image file is empty (0 bytes): {path}")

    img_bytes = path.read_bytes()
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"OpenCV could not decode image: {path}")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class AgriPredictor:
    """
    High-level inference wrapper for AgriPerceiver VLM.

    Handles model loading, image preprocessing, generation, and output parsing.

    Args:
        checkpoint_path: Path to the specialist checkpoint (.pt file).
        llm_name: HuggingFace ID for the language model.
        vision_name: HuggingFace ID for the vision encoder.
        device: Target device ("cuda", "cpu", or specific "cuda:0").
        max_new_tokens: Maximum tokens to generate.
        use_lora: Whether the checkpoint includes LoRA weights.
        lora_config: Optional custom LoRA configuration dict.
    """

    def __init__(
        self,
        checkpoint_path: str,
        llm_name: str = DEFAULT_LLM,
        vision_name: str = DEFAULT_VISION,
        device: str = "cuda",
        max_new_tokens: int = 350,
        use_lora: bool = True,
        lora_config: Optional[dict] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.tokenizer.add_tokens(SPECIAL_TOKENS)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Backbones
        phi3 = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
            attn_implementation="eager",
        ).to(self.device)
        phi3.resize_token_embeddings(len(self.tokenizer))

        siglip = AutoModel.from_pretrained(vision_name, torch_dtype=torch.bfloat16).to(self.device)

        # Model
        self.model = AgriPerceiverVLM(siglip, phi3, self.tokenizer).to(self.device)

        # LoRA (for Stage 2 specialist checkpoints)
        if use_lora:
            from peft import LoraConfig, get_peft_model

            if lora_config is None:
                lora_config = {
                    "r": 32,
                    "lora_alpha": 64,
                    "target_modules": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj", "up_proj"],
                    "lora_dropout": 0.1,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                }
            peft_config = LoraConfig(**lora_config)
            self.model.language_model = get_peft_model(self.model.language_model, peft_config)

        # Load weights
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(torch.bfloat16)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        image_path: Union[str, Path],
        prompt: str = SPECIALIST_PROMPT,
        return_raw: bool = False,
    ) -> Union[DiagnosticReport, dict, str]:
        """
        Run inference on a single image.

        Args:
            image_path: Path to the leaf image.
            prompt: Chat-formatted prompt (must contain <image> token).
            return_raw: If True, return raw generated text instead of parsed report.

        Returns:
            DiagnosticReport (or raw text if return_raw=True, or dict if parsing fails).
        """
        img = load_image(image_path)
        pixel_values = tile_image(img).unsqueeze(0).to(self.device, dtype=torch.bfloat16)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        visual_latents = self.model.encode_images(pixel_values)

        input_ids = inputs.input_ids
        curr_mask = inputs.attention_mask.to(torch.long)

        generated_tokens = []
        for _ in range(self.max_new_tokens):
            outputs, _ = self.model.splice_and_forward(input_ids, curr_mask, visual_latents)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.to(self.device)], dim=1)
            curr_mask = torch.cat([curr_mask, torch.ones((1, 1), device=self.device, dtype=torch.long)], dim=1)

        raw_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if return_raw:
            return raw_text

        report = DiagnosticReport.from_json_string(raw_text)
        if report is not None:
            return report

        # Fallback: return as dict with raw text
        return {"raw_output": raw_text, "parse_error": True}

    @torch.no_grad()
    def batch_predict(self, image_paths: list[Union[str, Path]], **kwargs) -> list:
        """Run predict on multiple images sequentially."""
        return [self.predict(p, **kwargs) for p in image_paths]


def main():
    """CLI entrypoint: agri-predict --image <path> --checkpoint <path>"""
    parser = argparse.ArgumentParser(description="AgriPerceiver VLM — Leaf Pathology Diagnosis")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to specialist checkpoint")
    parser.add_argument("--llm", type=str, default=DEFAULT_LLM, help="HuggingFace LLM identifier")
    parser.add_argument("--vision", type=str, default=DEFAULT_VISION, help="HuggingFace vision encoder identifier")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--max-tokens", type=int, default=350, help="Max generation tokens")
    parser.add_argument("--raw", action="store_true", help="Print raw text instead of parsed JSON")
    parser.add_argument("--no-lora", action="store_true", help="Skip LoRA (for Stage 1 checkpoints)")
    args = parser.parse_args()

    predictor = AgriPredictor(
        checkpoint_path=args.checkpoint,
        llm_name=args.llm,
        vision_name=args.vision,
        device=args.device,
        max_new_tokens=args.max_tokens,
        use_lora=not args.no_lora,
    )

    result = predictor.predict(args.image, return_raw=args.raw)

    if isinstance(result, str):
        print(result)
    elif isinstance(result, DiagnosticReport):
        print(json.dumps(result.model_dump(), indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
