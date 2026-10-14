import os
import sys
from dotenv import load_dotenv

# Ensure safe printing on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from legal_llm_client import (
    PROVIDERS_CONFIG,
    detect_available_providers,
    detect_default_provider,
    generate_chat_completion,
    get_friendly_error_message,
    LLMException,
)

load_dotenv()

def main():
    print("=" * 60)
    print("⚖️ LawBot LLM Provider Diagnostic Test")
    print("=" * 60)

    # 1. Detect keys
    available = detect_available_providers()
    print("\n[1] Environment API Key Status:")
    for prov_id, key in available.items():
        status = "✅ Configured (Protected)" if key else "❌ Not set"
        print(f"  • {PROVIDERS_CONFIG[prov_id]['name']}: {status}")

    # 2. Determine default
    default_prov, default_key = detect_default_provider()
    print(f"\n[2] Active Default Provider: {PROVIDERS_CONFIG[default_prov]['name']}")

    if not default_key:
        print("\n⚠️ No API key found for default provider!")
        print("Tip: Add a free GROQ_API_KEY in .env or Render dashboard.")
        print("Get your free key in 30 seconds at: https://console.groq.com/keys")
        return

    # 3. Test chat completion
    test_prompt = "In 2 sentences, explain Article 21 of the Constitution of India."
    print(f"\n[3] Testing chat completion with {PROVIDERS_CONFIG[default_prov]['name']}...")
    print(f"    Model: {PROVIDERS_CONFIG[default_prov]['default_model']}")
    print(f"    Prompt: \"{test_prompt}\"")

    try:
        ans = generate_chat_completion(
            prompt=test_prompt,
            provider=default_prov,
            api_key=default_key,
            max_tokens=150,
        )
        print("\n✅ Response Received Successfully:")
        print("-" * 50)
        print(ans)
        print("-" * 50)
        print("\n🎉 Provider is online and ready for LawBot!")
    except LLMException as e:
        friendly = get_friendly_error_message(e, default_prov)
        print(f"\n❌ {friendly['title']}")
        print(f"Details: {friendly['message']}")
        print(f"Action: {friendly['action']}")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")

if __name__ == "__main__":
    main()
