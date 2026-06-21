import asyncio
from types import SimpleNamespace

import httpx
import pytest

from app.services import llm_client as llm_client_module
from app.services.llm_client import LlmClient
from app.services.structured_llm import StructuredLlmValidationError, generate_validated_json


class FakeStructuredClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0
        self.user_prompts: list[str] = []
        self.response_formats: list[dict | None] = []

    async def generate_required(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict | None = None,
    ) -> str:
        self.calls += 1
        self.user_prompts.append(user_prompt)
        self.response_formats.append(response_format)
        return self.outputs[min(self.calls - 1, len(self.outputs) - 1)]


class FakeHttpResponse:
    def __init__(self, status_code: int = 200, content: str = "ok", text: str = "") -> None:
        self.status_code = status_code
        self._content = content
        self.text = text

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


def configured_llm_settings(**overrides) -> SimpleNamespace:
    values = {
        "llm_api_key": "key",
        "llm_base_url": "https://example.test",
        "llm_model": "demo-model",
        "llm_timeout_seconds": 3,
        "llm_max_retries": 2,
        "llm_retry_backoff_seconds": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def install_fake_async_client(monkeypatch: pytest.MonkeyPatch, outcomes: list[object]) -> list[dict]:
    calls: list[dict] = []

    class FakeAsyncClient:
        def __init__(self, timeout: int) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, headers: dict, json: dict):
            calls.append({"url": url, "headers": headers, "json": json, "timeout": self.timeout})
            outcome = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    monkeypatch.setattr(llm_client_module.httpx, "AsyncClient", FakeAsyncClient)
    return calls


def test_generate_validated_json_retries_and_passes_response_format() -> None:
    client = FakeStructuredClient(['{"value": 1}', '{"value": "ok"}'])

    data = asyncio.run(
        generate_validated_json(
            client,
            "只输出 JSON",
            '{"task":"demo"}',
            validate=lambda value: [] if isinstance(value.get("value"), str) else ["value 必须是字符串。"],
            error_message="bad json",
            response_format={"type": "json_object"},
        )
    )

    assert data == {"value": "ok"}
    assert client.calls == 2
    assert client.response_formats == [{"type": "json_object"}, {"type": "json_object"}]
    assert "value 必须是字符串" in client.user_prompts[1]


def test_generate_validated_json_raises_after_retry_failure() -> None:
    client = FakeStructuredClient(['{"value": 1}', '{"value": 2}'])

    with pytest.raises(StructuredLlmValidationError) as exc_info:
        asyncio.run(
            generate_validated_json(
                client,
                "只输出 JSON",
                '{"task":"demo"}',
                validate=lambda value: [] if isinstance(value.get("value"), str) else ["value 必须是字符串。"],
                error_message="bad json",
                response_format={"type": "json_object"},
            )
        )

    assert exc_info.value.errors == ["value 必须是字符串。"]
    assert exc_info.value.data == {"value": 2}
    assert client.calls == 2


def test_llm_client_sends_response_format_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads: list[dict] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    class FakeAsyncClient:
        def __init__(self, timeout: int) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, headers: dict, json: dict) -> FakeResponse:
            payloads.append(json)
            return FakeResponse()

    monkeypatch.setattr(llm_client_module.httpx, "AsyncClient", FakeAsyncClient)
    client = LlmClient()
    client.settings = configured_llm_settings()

    content = asyncio.run(
        client.generate_required(
            "system",
            "user",
            response_format={"type": "json_object"},
        )
    )

    assert content == '{"ok": true}'
    assert payloads[0]["response_format"] == {"type": "json_object"}


def test_llm_client_adds_deepseek_thinking_toggle_only_for_deepseek(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads: list[dict] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    class FakeAsyncClient:
        def __init__(self, timeout: int) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, headers: dict, json: dict) -> FakeResponse:
            payloads.append(json)
            return FakeResponse()

    monkeypatch.setattr(llm_client_module.httpx, "AsyncClient", FakeAsyncClient)

    deepseek_client = LlmClient()
    deepseek_client.settings = configured_llm_settings(
        llm_base_url="https://api.deepseek.com",
        llm_model="deepseek-v4-flash",
        llm_thinking_type="disabled",
    )
    asyncio.run(deepseek_client.generate_required("system", "user"))

    other_client = LlmClient()
    other_client.settings = configured_llm_settings(
        llm_base_url="https://example.test",
        llm_model="demo-model",
        llm_thinking_type="disabled",
    )
    asyncio.run(other_client.generate_required("system", "user"))

    assert payloads[0]["thinking"] == {"type": "disabled"}
    assert "thinking" not in payloads[1]


def test_llm_client_retries_timeout_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = install_fake_async_client(
        monkeypatch,
        [httpx.TimeoutException("request timed out"), FakeHttpResponse(content="after retry")],
    )
    client = LlmClient()
    client.settings = configured_llm_settings()

    content = asyncio.run(client.generate_required("system", "user"))

    assert content == "after retry"
    assert len(calls) == 2
    assert client.last_failure is None


def test_llm_client_retries_429_and_preserves_response_format(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = install_fake_async_client(
        monkeypatch,
        [
            FakeHttpResponse(status_code=429, text="rate limited"),
            FakeHttpResponse(content='{"ok": true}'),
        ],
    )
    client = LlmClient()
    client.settings = configured_llm_settings()

    content = asyncio.run(
        client.generate_required(
            "system",
            "user",
            response_format={"type": "json_object"},
        )
    )

    assert content == '{"ok": true}'
    assert len(calls) == 2
    assert calls[0]["json"]["response_format"] == {"type": "json_object"}
    assert calls[1]["json"]["response_format"] == {"type": "json_object"}


def test_llm_client_retries_503_twice_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = install_fake_async_client(
        monkeypatch,
        [
            FakeHttpResponse(status_code=503, text="busy 1"),
            FakeHttpResponse(status_code=503, text="busy 2"),
            FakeHttpResponse(content="ok after 503"),
        ],
    )
    client = LlmClient()
    client.settings = configured_llm_settings()

    content = asyncio.run(client.generate_required("system", "user"))

    assert content == "ok after 503"
    assert len(calls) == 3


@pytest.mark.parametrize("status_code", [400, 401, 403])
def test_llm_client_does_not_retry_nonretryable_http_status(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    calls = install_fake_async_client(
        monkeypatch,
        [FakeHttpResponse(status_code=status_code, text="bad request")],
    )
    client = LlmClient()
    client.settings = configured_llm_settings()

    content = asyncio.run(client.generate("system", "user"))

    assert content is None
    assert len(calls) == 1
    assert client.last_failure is not None
    assert client.last_failure.status_code == status_code


def test_llm_client_generate_required_reports_exhausted_retry_details(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = install_fake_async_client(
        monkeypatch,
        [
            FakeHttpResponse(status_code=429, text="rate limited 1"),
            FakeHttpResponse(status_code=500, text="server failed 2"),
            FakeHttpResponse(status_code=429, text="rate limited final"),
        ],
    )
    client = LlmClient()
    client.settings = configured_llm_settings()

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(client.generate_required("system", "user"))

    message = str(exc_info.value)
    assert len(calls) == 3
    assert "attempt=3/3" in message
    assert "status_code=429" in message
    assert "rate limited final" in message


def test_llm_client_unconfigured_does_not_create_http_client(monkeypatch: pytest.MonkeyPatch) -> None:
    created_clients = 0

    class FakeAsyncClient:
        def __init__(self, timeout: int) -> None:
            nonlocal created_clients
            created_clients += 1

    monkeypatch.setattr(llm_client_module.httpx, "AsyncClient", FakeAsyncClient)
    client = LlmClient()
    client.settings = configured_llm_settings(llm_api_key=None)

    content = asyncio.run(client.generate("system", "user"))

    assert content is None
    assert created_clients == 0
    assert client.last_failure is not None
    assert client.last_failure.error_type == "not_configured"
