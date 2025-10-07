
from openai import OpenAI, APITimeoutError
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

def ask_openrouter(prompt: str, model: str = "qwen/qwen3-235b-a22b:free") -> str:
    """Send a prompt to OpenRouter and return the model's response."""
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:8000",  # update later with your site URL
                "X-Title": "Model Validator",
            },
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=30.0,  # Add a 30-second timeout
        )
        return completion.choices[0].message.content
    except APITimeoutError:
        return "The request to the AI service timed out. Please try again later."
    except Exception as e:
        return f"An error occurred while communicating with the AI service: {str(e)}"