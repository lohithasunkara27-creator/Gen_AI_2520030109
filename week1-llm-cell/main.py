

import os
from dotenv import load_dotenv
from groq import Groq

# Step 1: Load environment variables
load_dotenv()

# Step 2: Read API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Groq API key not found!")

# Step 3: Initialize Groq client
client = Groq(api_key=api_key)

# Step 4: Send prompt to AI
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "What is Artificial Intelligence? Explain in simple words."
        }
    ]
)

# Step 5: Display response
print("\nLLM Response:")
print(response.choices[0].message.content)
