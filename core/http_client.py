"""Small dependency-free HTTP transport used by external service adapters."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class HttpClientError(RuntimeError):
    """Raised when an external HTTP request cannot be completed."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class UrllibTransport:
    """JSON/text HTTP transport with bounded retries for transient failures."""

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 2,
        backoff_seconds: float = 0.5,
        user_agent: str = "AcademicResearchAssistant/1.0",
    ) -> None:
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.backoff_seconds = max(0.0, backoff_seconds)
        self.user_agent = user_agent

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        return json.loads(self.get_text(url, params=params, headers=headers))

    def get_text(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        return self._request("GET", url, params=params, headers=headers).decode(
            "utf-8", errors="replace"
        )

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> Any:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        response = self._request(
            "POST",
            url,
            headers={"Content-Type": "application/json", **(headers or {})},
            body=body,
        )
        try:
            return json.loads(response)
        except json.JSONDecodeError as exc:
            raise HttpClientError(f"响应不是有效 JSON: {url}") from exc

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ) -> bytes:
        if params:
            query = urlencode(params, doseq=True)
            url = f"{url}{'&' if '?' in url else '?'}{query}"

        request_headers = {
            "Accept": "application/json, application/xml, text/plain",
            "User-Agent": self.user_agent,
            **(headers or {}),
        }

        for attempt in range(self.max_retries + 1):
            request = Request(
                url,
                data=body,
                headers=request_headers,
                method=method,
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return response.read()
            except HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt >= self.max_retries:
                    detail = exc.read().decode("utf-8", errors="replace").strip()
                    if detail:
                        detail = " ".join(detail.split())[:1_000]
                    raise HttpClientError(
                        f"HTTP {exc.code} from {url}{': ' + detail if detail else ''}",
                        status_code=exc.code,
                    ) from exc
            except (URLError, TimeoutError) as exc:
                if attempt >= self.max_retries:
                    raise HttpClientError(f"请求失败: {url}: {exc}") from exc

            time.sleep(self.backoff_seconds * (2**attempt))

        raise HttpClientError(f"请求失败: {url}")
