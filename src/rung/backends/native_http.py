"""Minimal native HTTP transports: no dependency changes to the frozen simulator.

Credentials stay in headers, never request archives, command lines or exception text.
One POST per call, no transparent retries or fallback model.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request

from rung.backends.base import BackendError


class NativeHTTP:
    def __init__(self, endpoint: str, headers: dict):
        self.endpoint, self.headers = endpoint, headers

    def create(self, *, timeout: float = 90, **body):
        from rung.backends.native_session import encode_body

        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()
        request = urllib.request.Request(
            self.endpoint,
            encode_body(body),
            {"Content-Type": "application/json", **self.headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise BackendError(f"native provider HTTP {exc.code}; no retry") from None
        except urllib.error.URLError as exc:
            raise BackendError(f"native transport {type(exc.reason).__name__}; no retry") from None


class GeminiHTTPClient:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Gemini API credential missing")
        self.interactions = NativeHTTP(
            "https://generativelanguage.googleapis.com/v1beta/interactions", {"x-goog-api-key": key}
        )


class OpenAIHTTPClient:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API credential missing")
        self.responses = NativeHTTP(
            "https://api.openai.com/v1/responses", {"Authorization": f"Bearer {key}"}
        )
