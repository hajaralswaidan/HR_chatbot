from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LocalQwen:
    """
    Lightweight local model (CPU).
    Default model is small Falcon 1B to avoid overloading the device.
    """

    def __init__(self, model_name: str = "tiiuae/falcon-rw-1b"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load(self):
        if self.model is not None:
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="cpu",
            torch_dtype=torch.float32,
        )

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
