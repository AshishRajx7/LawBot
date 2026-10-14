# tests/test_legal_llm_client.py — Unit Tests for Multi-Provider LLM Client
import unittest
from unittest.mock import patch, MagicMock
import os

from legal_llm_client import (
    PROVIDERS_CONFIG,
    detect_available_providers,
    detect_default_provider,
    get_friendly_error_message,
    generate_chat_completion,
    LLMQuotaExhaustedError,
    LLMRateLimitError,
    LLMAuthError,
    LLMProviderError,
)


class TestLegalLLMClient(unittest.TestCase):

    def test_providers_configuration_integrity(self):
        """Verify all supported providers have required metadata."""
        for prov_id, cfg in PROVIDERS_CONFIG.items():
            self.assertIn("name", cfg)
            self.assertIn("base_url", cfg)
            self.assertIn("default_model", cfg)
            self.assertIn("env_keys", cfg)
            self.assertTrue(cfg["base_url"].startswith("http"))

    @patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test123", "HF_API_KEY": "hf_test456"}, clear=True)
    def test_detect_default_provider_prioritizes_groq(self):
        """Verify Groq is prioritized over HF if GROQ_API_KEY is present."""
        prov, key = detect_default_provider()
        self.assertEqual(prov, "groq")
        self.assertEqual(key, "gsk_test123")

    @patch.dict(os.environ, {"HF_API_KEY": "hf_test456"}, clear=True)
    def test_detect_default_provider_falls_back_to_hf(self):
        """Verify HF is detected when only HF_API_KEY is present."""
        prov, key = detect_default_provider()
        self.assertEqual(prov, "huggingface")
        self.assertEqual(key, "hf_test456")

    @patch.dict(os.environ, {}, clear=True)
    def test_detect_default_provider_without_env(self):
        """Verify fallback to groq when no keys are configured."""
        prov, key = detect_default_provider()
        self.assertEqual(prov, "groq")
        self.assertIsNone(key)

    def test_missing_api_key_raises_auth_error(self):
        """Verify calling generate_chat_completion without key raises LLMAuthError."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LLMAuthError):
                generate_chat_completion(
                    prompt="Hello",
                    provider="groq",
                    api_key=None
                )

    @patch("requests.post")
    def test_successful_chat_completion(self, mock_post):
        """Verify parsing standard 200 OK OpenAI-compatible response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {"message": {"role": "assistant", "content": "Article 21 guarantees life and personal liberty."}}
            ]
        }
        mock_post.return_value = mock_resp

        ans = generate_chat_completion(
            prompt="Explain Article 21",
            provider="groq",
            api_key="gsk_valid_key"
        )
        self.assertIn("Article 21 guarantees", ans)
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_402_payment_required_raises_quota_exhausted_error(self, mock_post):
        """Verify HTTP 402 is translated to LLMQuotaExhaustedError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 402
        mock_resp.text = "You have no remaining credits. Purchase pre-paid credits."
        mock_resp.json.return_value = {"error": "You have no remaining credits."}
        mock_post.return_value = mock_resp

        with self.assertRaises(LLMQuotaExhaustedError) as ctx:
            generate_chat_completion(
                prompt="Explain Article 21",
                provider="huggingface",
                api_key="hf_empty_credits"
            )
        self.assertEqual(ctx.exception.status_code, 402)
        self.assertEqual(ctx.exception.provider, "huggingface")

        # Verify friendly error translation
        msg = get_friendly_error_message(ctx.exception, "huggingface")
        self.assertIn("Credits Exhausted", msg["title"])
        self.assertIn("Groq", msg["action"])

    @patch("requests.post")
    def test_429_rate_limit_raises_rate_limit_error(self, mock_post):
        """Verify HTTP 429 is translated to LLMRateLimitError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "Rate limit reached."
        mock_resp.json.return_value = {"error": "Rate limit reached."}
        mock_post.return_value = mock_resp

        with self.assertRaises(LLMRateLimitError) as ctx:
            generate_chat_completion(
                prompt="Explain Article 21",
                provider="groq",
                api_key="gsk_key"
            )
        self.assertEqual(ctx.exception.status_code, 429)

        msg = get_friendly_error_message(ctx.exception, "groq")
        self.assertIn("Rate Limit", msg["title"])

    @patch("requests.post")
    def test_401_unauthorized_raises_auth_error(self, mock_post):
        """Verify HTTP 401 is translated to LLMAuthError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Invalid API Key"
        mock_resp.json.return_value = {"error": "Invalid API Key"}
        mock_post.return_value = mock_resp

        with self.assertRaises(LLMAuthError) as ctx:
            generate_chat_completion(
                prompt="Explain Article 21",
                provider="openrouter",
                api_key="invalid_key"
            )
        self.assertEqual(ctx.exception.status_code, 401)


if __name__ == "__main__":
    unittest.main()
