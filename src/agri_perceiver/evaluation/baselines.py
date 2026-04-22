"""
Baseline model runners for comparative evaluation.

Wraps Gemma-3, LLaVA-NeXT, and InternVL2 behind a uniform interface.
Each baseline takes an image path and returns a JSON diagnostic string.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaselineModel(ABC):
    """Abstract interface for baseline VLM comparisons."""

    name: str

    @abstractmethod
    def predict(self, image_path: str, prompt: str) -> str:
        """Return raw JSON string prediction for the given image."""

    def batch_predict(self, image_paths: list[str], prompt: str) -> list[str]:
        return [self.predict(p, prompt) for p in image_paths]


BASELINE_PROMPT = (
    "You are an agricultural pathology expert. Analyze this leaf image and "
    "provide a structured diagnostic report in JSON format with these fields: "
    "diagnosis, type (fungal/bacterial/viral/pest/deficiency/unknown), "
    "severity (0-1), confidence (0-1), symptoms (list), reasoning (string), "
    "recommended_actions (list). Return ONLY valid JSON."
)


class Qwen2VLBaseline(BaselineModel):
    """Qwen2-VL-7B-Instruct baseline — strong open VLM, non-gated, 7B params."""

    name = "Qwen2-VL-7B"

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct", device: str = "cuda"):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()

    def predict(self, image_path: str, prompt: str = BASELINE_PROMPT) -> str:
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ]}
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512)
        generated = outputs[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]


class LLaVANextBaseline(BaselineModel):
    """LLaVA-NeXT baseline via transformers."""

    name = "LLaVA-NeXT-7B"

    def __init__(self, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf", device: str = "cuda"):
        import torch
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

        self.device = device
        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

    def predict(self, image_path: str, prompt: str = BASELINE_PROMPT) -> str:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        return self.processor.decode(outputs[0], skip_special_tokens=True)


class InternVL2Baseline(BaselineModel):
    """InternVL2 baseline."""

    name = "InternVL2-8B"

    def __init__(self, model_name: str = "OpenGVLab/InternVL2-8B", device: str = "cuda"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().to(device)

        # Fix for transformers >= 4.50: GenerationMixin was decoupled from PreTrainedModel.
        # InternLM2ForCausalLM in the cached trust_remote_code relies on the old inheritance
        # where PreTrainedModel included GenerationMixin. Inject it back into the class MRO
        # so that language_model.generate() is available for InternVLChatModel.chat().
        from transformers.generation.utils import GenerationMixin
        lm_class = type(self.model.language_model)
        if not isinstance(self.model.language_model, GenerationMixin):
            lm_class.__bases__ = lm_class.__bases__ + (GenerationMixin,)

    def predict(self, image_path: str, prompt: str = BASELINE_PROMPT) -> str:
        from PIL import Image
        import torchvision.transforms as T

        image = Image.open(image_path).convert("RGB")
        pixel_values = self._preprocess(image).unsqueeze(0).to(self.device)

        generation_config = {"max_new_tokens": 512, "do_sample": False}
        response = self.model.chat(self.tokenizer, pixel_values, prompt, generation_config)
        return response

    def _preprocess(self, image):
        import torch
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((448, 448)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).to(torch.bfloat16)


# Registry for easy access
BASELINES = {
    "qwen2vl": Qwen2VLBaseline,
    "llava_next": LLaVANextBaseline,
    "internvl2": InternVL2Baseline,
}


def get_baseline(name: str, **kwargs) -> BaselineModel:
    """Instantiate a baseline model by name."""
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline '{name}'. Available: {list(BASELINES.keys())}")
    return BASELINES[name](**kwargs)
