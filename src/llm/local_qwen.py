import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_MODEL = "Qwen/Qwen2-1.5B-Instruct"


class LocalQwen:
    """
    Local LLM (Qwen2-1.5B-Instruct) for Text-to-SQL.
    - CPU friendly نسبياً
    - Deterministic (do_sample=False) لتقليل الهبد في SQL
    - يرجّع نص فقط (والـ sql_agent يسوي extract/select)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None

    def load(self):
        if self.model is not None and self.tokenizer is not None:
            return

        print(f" Loading Local Qwen: {self.model_name} (CPU)...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="cpu",
            torch_dtype=torch.float32
        )

        self.model.eval()
        print("Qwen Loaded (CPU)")

    def _clean_output(self, text: str, prompt: str) -> str:
        t = (text or "").strip()

        if prompt and prompt in t:
            t = t.split(prompt, 1)[-1].strip()

        for key in ["SQL:", "Sql:", "sql:", "Assistant:", "Answer:", "OUTPUT:", "Output:"]:
            if key in t:
                t = t.split(key)[-1].strip()

       
        t = re.sub(r"```sql|```", "", t, flags=re.I).strip()

        return t

    def generate(self, prompt: str, max_new_tokens: int = 192) -> str:
        self.load()
        assert self.tokenizer is not None and self.model is not None

        inputs = self.tokenizer(prompt, return_tensors="pt")
        # inputs on CPU
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,     
                temperature=0.0,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._clean_output(decoded, prompt)
