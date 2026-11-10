import urllib.error

import pytest

from rung.backends.base import BackendError
from rung.backends.native_http import NativeHTTP


def test_server_failure_is_one_request_without_secret_in_error(monkeypatch):
    calls = []

    def fail(request, **kwargs):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 500, "server error", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(BackendError, match="HTTP 500; no retry") as exc:
        NativeHTTP("https://example.invalid", {"x-goog-api-key": "test-secret"}).create(
            model="test"
        )
    assert len(calls) == 1
    assert "test-secret" not in str(exc.value)
