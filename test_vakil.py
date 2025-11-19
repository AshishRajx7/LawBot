# test_mistral_direct.py
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load API key
load_dotenv()
hf_token = os.getenv("HF_API_KEY")

# Direct client
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=hf_token,
)

prompt = """Explain Article 21 of the Indian Constitution in simple legal terms."""

print("⏳ Querying model...")

response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are LawBot, an expert in Indian constitutional law."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=300,
)

print("\n✅ Model connected successfully!\n")
print("🔹 Response:\n", response.choices[0].message["content"])
