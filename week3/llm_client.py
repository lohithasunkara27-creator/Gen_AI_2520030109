
import time
import requests
from dataclasses import dataclass


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int


class OllamaProvider:

    provider_name = "Ollama"

    def __init__(self, model="llama3.2"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt):

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()
        data = response.json()

        return LLMResult(
            text=data.get("response", ""),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0)
        )


class LLMClient:

    def __init__(self, provider, usage_logger):
        self.provider = provider
        self.usage_logger = usage_logger

    def generate(self, prompt):

        start = time.perf_counter()

        for attempt in range(3):
            try:
                result = self.provider.generate(prompt)

                latency = round(
                    (time.perf_counter() - start) * 1000, 2
                )

                self.usage_logger.log_call({
                    "provider": "Ollama",
                    "model": self.provider.model,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "total_tokens": (
                        result.input_tokens + result.output_tokens
                    ),
                    "latency_ms": latency,
                    "estimated_cost_usd": 0.0,
                    "status": "success",
                    "attempts": attempt + 1
                })

                print("Latency:", latency, "ms")
                print("Input tokens:", result.input_tokens)
                print("Output tokens:", result.output_tokens)

                return result.text

            except requests.exceptions.RequestException as error:

                if attempt == 2:
                    self.usage_logger.log_call({
                        "provider": "Ollama",
                        "model": self.provider.model,
                        "status": "failed",
                        "attempts": attempt + 1,
                        "error": str(error)
                    })
                    raise

                delay = 2 ** attempt
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)

