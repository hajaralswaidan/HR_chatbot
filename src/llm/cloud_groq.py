from groq import Groq
import os


class CloudGroq:
    """
    Cloud LLM using Groq API (for Text-to-SQL).
    """

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.model = model

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set.")

        self.client = Groq(api_key=api_key)

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a SQLite SQL assistant. Return ONLY JSON."
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                top_p=0.9,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content.strip()

        except Exception as e:
            return f'{{"error":"Groq failed: {e}"}}'
