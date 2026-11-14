
from dotenv import load_dotenv

from llm_client import OllamaProvider, LLMClient
from usage_logger import UsageLogger

load_dotenv(override=True)


def main():

    logger = UsageLogger()

    try:
        provider = OllamaProvider("llama3.2")
        client = LLMClient(provider, logger)

        print("\n===== LLM MONITORING WITH MONGODB =====")
        print("Provider: Ollama")
        print("Model: llama3.2")

        prompt = input("\nEnter your question: ")

        if prompt.strip():
            answer = client.generate(prompt)

            print("\n===== RESPONSE =====")
            print(answer)

    finally:
        logger.close()


if __name__ == "__main__":
    main()

