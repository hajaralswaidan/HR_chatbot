# src/llm/local_qwen.py
from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config import (
    LOCAL_MODEL_PATH,
    LOCAL_MAX_NEW_TOKENS,
    LOCAL_TEMPERATURE,
    LOCAL_TOP_P,
)

class LocalQwen:
    """
    Local Qwen runner (no external API).
    Loads model/tokenizer once and generates answers.
    """

    def __init__(self, model_path=LOCAL_MODEL_PATH):
        self.model_path = str(model_path)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Recommended settings for limited RAM/VRAM
        # (If you have GPU, it will use it automatically)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            local_files_only=True,
        )

        if self.device == "cpu":
            self.model.to(self.device)

        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=LOCAL_MAX_NEW_TOKENS,
            temperature=LOCAL_TEMPERATURE,
            top_p=LOCAL_TOP_P,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Return only assistant completion (best-effort)
        # If prompt is included at start, remove it.
        if text.startswith(prompt):
            text = text[len(prompt):].strip()

        return text.strip()
