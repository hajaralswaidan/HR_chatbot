import os
import streamlit as st
from huggingface_hub import InferenceClient


class CloudQwen:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-72B-Instruct"):
        self.model_id = model_id
        self._client = None
        self._token_cache = None

    def _get_token(self) -> str:
        try:
            token = (st.secrets.get("HF_TOKEN", "") or "").strip()
            if token:
                return token
        except Exception:
            pass

        return os.getenv("HF_TOKEN", "").strip()

    def is_configured(self) -> bool:
        return bool(self._get_token())

    def _get_client(self) -> InferenceClient:
        token = self._get_token()
        if not token:
            raise RuntimeError("HF_TOKEN is missing. Add it to .streamlit/secrets.toml")

        if self._client is None or self._token_cache != token:
            self._client = InferenceClient(token=token)
            self._token_cache = token

        return self._client

    def generate(self, prompt: str, max_new_tokens: int = 300, temperature: float = 0.2) -> str:
        client = self._get_client()

        resp = client.chat_completion(
            model=self.model_id,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful HR assistant. "
                        "Use the provided context only and do not invent numbers."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_new_tokens,
            temperature=temperature,
            stream=False,
        )

        return (resp.choices[0].message.content or "").strip()
