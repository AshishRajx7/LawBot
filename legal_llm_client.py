# legal_llm_client.py — Production Multi-Provider LLM Client for LawBot
# Supports Groq (Free Tier), Hugging Face, OpenRouter, Google Gemini & Custom OpenAI-compatible endpoints.
import os
import json
import logging
from typing import Dict, Any, Optional, Tuple
import requests

logger = logging.getLogger("lawbot.llm")

# ==============================================================================
# 1. PROVIDER CONFIGURATION REGISTRY
# ==============================================================================

PROVIDERS_CONFIG: Dict[str, Dict[str, Any]] = {
    "groq": {
        "name": "Groq",
        "display_name": "⚡ Groq (Recommended — 100% Free & Ultra-Fast)",
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "default_model": "llama-3.1-8b-instant",
        "supported_models": [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
        ],
        "env_keys": ["GROQ_API_KEY"],
        "signup_url": "https://console.groq.com/keys",
        "is_free": True,
        "notes": "14,400 requests/day free tier. Sub-second response times.",
    },
    "huggingface": {
        "name": "Hugging Face",
        "display_name": "🤗 Hugging Face (Inference Router)",
        "base_url": "https://router.huggingface.co/v1/chat/completions",
        "default_model": "meta-llama/Llama-3.1-8B-Instruct",
        "supported_models": [
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "Qwen/Qwen2.5-7B-Instruct",
        ],
        "env_keys": ["HF_API_KEY", "HUGGINGFACE_API_KEY", "HF_TOKEN"],
        "signup_url": "https://huggingface.co/settings/tokens",
        "is_free": False,
        "notes": "Requires pre-paid credits or PRO subscription for Llama-3.1 models.",
    },
    "openrouter": {
        "name": "OpenRouter",
        "display_name": "🌐 OpenRouter (Free & Pay-as-you-go Models)",
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "default_model": "meta-llama/llama-3.1-8b-instruct:free",
        "supported_models": [
            "meta-llama/llama-3.1-8b-instruct:free",
            "google/gemini-2.0-flash-exp:free",
            "mistralai/mistral-7b-instruct:free",
        ],
        "env_keys": ["OPENROUTER_API_KEY"],
        "signup_url": "https://openrouter.ai/keys",
        "is_free": True,
        "notes": "Free models available without payment method.",
    },
    "gemini": {
        "name": "Google Gemini",
        "display_name": "✨ Google Gemini (Google AI Studio Free Tier)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "default_model": "gemini-1.5-flash",
        "supported_models": [
            "gemini-1.5-flash",
            "gemini-2.0-flash",
        ],
        "env_keys": ["GEMINI_API_KEY"],
        "signup_url": "https://aistudio.google.com/app/apikey",
        "is_free": True,
        "notes": "Generous free tier via Google AI Studio.",
    },
    "custom": {
        "name": "Custom OpenAI-Compatible",
        "display_name": "⚙️ Custom OpenAI-Compatible (Ollama, vLLM, OpenAI)",
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"),
        "default_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "supported_models": ["custom"],
        "env_keys": ["OPENAI_API_KEY"],
        "signup_url": "",
        "is_free": False,
        "notes": "Any endpoint following OpenAI /v1/chat/completions specification.",
    },
}


# ==============================================================================
# 2. CUSTOM TYPED EXCEPTIONS FOR GRACEFUL UI RENDERING
# ==============================================================================

class LLMException(Exception):
    """Base exception for LLM inference errors."""
    def __init__(self, message: str, provider: str = "", status_code: Optional[int] = None, details: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.details = details or ""


class LLMQuotaExhaustedError(LLMException):
    """Raised when provider returns HTTP 402 or credits/quota are exhausted."""
    pass


class LLMRateLimitError(LLMException):
    """Raised when provider returns HTTP 429 (Rate limit reached)."""
    pass


class LLMAuthError(LLMException):
    """Raised when provider returns HTTP 401 or 403 (Invalid or missing API key)."""
    pass


class LLMProviderError(LLMException):
    """Raised when provider returns 5xx server errors or connection failures."""
    pass


# ==============================================================================
# 3. HELPER FUNCTIONS: PROVIDER DETECTION & FRIENDLY MESSAGES
# ==============================================================================

def detect_available_providers() -> Dict[str, Optional[str]]:
    """Check environment variables and return active keys for each provider."""
    found = {}
    for prov_id, cfg in PROVIDERS_CONFIG.items():
        key_found = None
        for env_name in cfg["env_keys"]:
            val = os.getenv(env_name, "").strip()
            if val:
                key_found = val
                break
        found[prov_id] = key_found
    return found


def detect_default_provider() -> Tuple[str, Optional[str]]:
    """
    Auto-detect the best available provider based on environment variables.
    Priority order:
    1. GROQ_API_KEY (Recommended: high reliability, ultra-fast, free tier)
    2. OPENROUTER_API_KEY
    3. GEMINI_API_KEY
    4. HF_API_KEY
    5. Fallback to groq with None key
    """
    active_keys = detect_available_providers()
    priority = ["groq", "openrouter", "gemini", "huggingface", "custom"]
    for prov in priority:
        if active_keys.get(prov):
            return prov, active_keys[prov]
    
    # If explicitly set in LLM_PROVIDER
    env_prov = os.getenv("LLM_PROVIDER", "").lower().strip()
    if env_prov in PROVIDERS_CONFIG:
        return env_prov, active_keys.get(env_prov)

    # Default to Groq
    return "groq", None


def get_friendly_error_message(err: Exception, provider: str) -> Dict[str, str]:
    """
    Format user-facing error message with actionable recovery guidance.
    Prevents exposing raw internal URLs, tokens, or scary stack traces.
    """
    prov_name = PROVIDERS_CONFIG.get(provider, {}).get("name", provider.title())
    signup_url = PROVIDERS_CONFIG.get(provider, {}).get("signup_url", "")
    
    if isinstance(err, LLMQuotaExhaustedError):
        if provider == "huggingface":
            return {
                "title": "💳 Hugging Face Inference Credits Exhausted",
                "message": (
                    "Hugging Face has migrated serverless models (Llama-3.1) to paid Inference Providers, "
                    "and your account currently has no remaining credits (HTTP 402)."
                ),
                "action": (
                    "**Immediate Free Solution**: Switch provider to **Groq** in the sidebar. "
                    "Groq offers a 100% free tier with 14,400 daily requests. "
                    "Alternatively, add pre-paid credits in your Hugging Face billing dashboard."
                ),
            }
        return {
            "title": f"💳 {prov_name} Credits or Quota Exhausted",
            "message": f"The inference quota for {prov_name} has been reached.",
            "action": f"Please switch to **Groq (Free)** in the sidebar or top up your {prov_name} account.",
        }

    if isinstance(err, LLMRateLimitError):
        return {
            "title": f"⏳ {prov_name} Rate Limit Reached",
            "message": f"The {prov_name} API is receiving high volume or your quota limit per minute was reached.",
            "action": "Please wait 10-15 seconds and try again, or switch to another provider in the sidebar.",
        }

    if isinstance(err, LLMAuthError):
        return {
            "title": f"🔑 {prov_name} Authentication Error",
            "message": f"The API key provided for {prov_name} is invalid, expired, or missing.",
            "action": (
                f"Enter a valid API key in the sidebar. "
                + (f"Get one for free at [{signup_url}]({signup_url})." if signup_url else "")
            ),
        }

    return {
        "title": f"⚠️ {prov_name} Service Notice",
        "message": "The LLM provider could not generate an answer at this moment.",
        "action": (
            "You can still inspect the retrieved precedents, ratio decidendi, and knowledge graph insights below. "
            "Try switching providers in the sidebar (e.g. Groq) or retry in a few moments."
        ),
    }


# ==============================================================================
# 4. CHAT COMPLETION EXECUTION
# ==============================================================================

def generate_chat_completion(
    prompt: str,
    provider: str = "groq",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 850,
    temperature: float = 0.2,
    timeout: int = 40,
) -> str:
    """
    Execute chat completion against an OpenAI-compatible endpoint (Groq, OpenRouter, HF, Gemini, Custom).
    Returns clean text answer or raises specific typed exceptions.
    """
    cfg = PROVIDERS_CONFIG.get(provider.lower())
    if not cfg:
        raise ValueError(f"Unsupported LLM provider: '{provider}'. Supported: {list(PROVIDERS_CONFIG.keys())}")

    active_model = model or cfg["default_model"]
    endpoint_url = cfg["base_url"]

    # Resolve API Key
    resolved_key = (api_key or "").strip()
    if not resolved_key:
        for env_k in cfg["env_keys"]:
            env_val = os.getenv(env_k, "").strip()
            if env_val:
                resolved_key = env_val
                break

    if not resolved_key:
        raise LLMAuthError(
            message=f"No API key provided for {cfg['name']}.",
            provider=provider,
            status_code=401,
        )

    headers = {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
    }

    # OpenRouter specific headers for attribution
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/AshishRajx7/LawBot"
        headers["X-Title"] = "LawBot Indian Constitutional Law Assistant"

    payload = {
        "model": active_model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(
            endpoint_url,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.exceptions.Timeout:
        raise LLMProviderError(
            message=f"Request to {cfg['name']} timed out after {timeout} seconds.",
            provider=provider,
        )
    except requests.exceptions.RequestException as e:
        raise LLMProviderError(
            message=f"Network error connecting to {cfg['name']}: {str(e)}",
            provider=provider,
        )

    status = response.status_code

    if status == 200:
        try:
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise LLMProviderError(f"No response choices returned by {cfg['name']}.", provider=provider)
            content = choices[0].get("message", {}).get("content", "")
            return content.strip()
        except Exception as e:
            raise LLMProviderError(f"Failed to parse response JSON from {cfg['name']}: {e}", provider=provider)

    # Handle Error Codes
    error_detail = response.text[:400]
    try:
        err_json = response.json()
        if "error" in err_json:
            if isinstance(err_json["error"], dict):
                error_detail = err_json["error"].get("message", error_detail)
            else:
                error_detail = str(err_json["error"])
    except Exception:
        pass

    logger.warning("LLM call to %s failed with status %d: %s", cfg["name"], status, error_detail)

    if status == 402:
        raise LLMQuotaExhaustedError(
            message=f"{cfg['name']} returned 402 Payment Required: {error_detail}",
            provider=provider,
            status_code=402,
            details=error_detail,
        )
    elif status == 429:
        raise LLMRateLimitError(
            message=f"{cfg['name']} returned 429 Rate Limit Exceeded: {error_detail}",
            provider=provider,
            status_code=429,
            details=error_detail,
        )
    elif status in (401, 403):
        raise LLMAuthError(
            message=f"{cfg['name']} returned {status} Unauthorized: {error_detail}",
            provider=provider,
            status_code=status,
            details=error_detail,
        )
    else:
        raise LLMProviderError(
            message=f"{cfg['name']} error (HTTP {status}): {error_detail}",
            provider=provider,
            status_code=status,
            details=error_detail,
        )
