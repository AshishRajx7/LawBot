from huggingface_hub import HfApi

import os
from dotenv import load_dotenv
load_dotenv()

api = HfApi()
key = os.getenv("HF_API_KEY")

if key:
    print("✅ Token loaded successfully!")
    user = api.whoami(token=key)
    print("Connected to:", user["name"])
else:
    print("❌ Token not found in .env")
