from groq import Groq
import os


class CloudGroq:
    """
    Cloud LLM using Groq API (recommended for this task).
    """

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.model = model
        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY")
        )

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        # api_key missing
        if not os.environ.get("GROQ_API_KEY"):
            return "ERROR: GROQ_API_KEY not set."

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            return f"ERROR: Groq failed: {e}"
