
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=".env.local")

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables or .env.local")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

def ask_openrouter(prompt: str, model: str = "mistralai/mistral-small-3.2-24b-instruct:free") -> str:
    """Send a prompt to OpenRouter and return the model's response."""
    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "http://localhost:8000",  # update later with your site URL
            "X-Title": "Model Validator",
        },
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content