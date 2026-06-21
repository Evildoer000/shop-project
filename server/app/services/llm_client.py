from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from collections.abc import AsyncGenerator
from typing import Any
from urllib.parse import urlparse

import httpx

from app.core.config import get_settings


logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# 设 LLM_DEBUG_LOG_PATH=/tmp/phase3.jsonl 把每次 LLM call 的 prompt+response 写盘
_PROMPT_LOG_PATH = os.environ.get("LLM_DEBUG_LOG_PATH", "")
_PROMPT_LOG_SEQ = {"n": 0}


def _log_prompt(component: str, operation: str, system_prompt: str, user_prompt: str,
                response: str | None, error: str = "", duration_ms: float = 0.0) -> None:
    if not _PROMPT_LOG_PATH:
        return
    _PROMPT_LOG_SEQ["n"] += 1
    record = {
        "seq": _PROMPT_LOG_SEQ["n"],
        "ts": time.strftime("%H:%M:%S"),
        "component": component,
        "operation": operation,
        "duration_ms": round(duration_ms, 1),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response": response,
        "error": error,
    }
    try:
        with open(_PROMPT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


@dataclass
class LlmCallFailure:
    error_type: str
    message: str
    attempt: int
    max_attempts: int
    status_code: int | None = None
    response_body_preview: str = ""

    def summary(self) -> str:
        parts = [
            f"error_type={self.error_type}",
            f"attempt={self.attempt}/{self.max_attempts}",
        ]
        if self.status_code is not None:
            parts.append(f"status_code={self.status_code}")
        if self.message:
            parts.append(f"message={self.message}")
        if self.response_body_preview:
            parts.append(f"response_body_preview={self.response_body_preview}")
        return ", ".join(parts)


@dataclass
class ModelCallTelemetry:
    component: str
    model: str
    provider: str
    operation: str
    stream: bool
    status: str
    latency_ms: float
    usage: dict[str, Any]
    estimated_cost: float | None = None
    attempt_count: int = 0
    error_type: str | None = None
    status_code: int | None = None

    def model_dump(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "component": self.component,
            "model": self.model,
            "provider": self.provider,
            "operation": self.operation,
            "stream": self.stream,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "usage": self.usage,
            "attempt_count": self.attempt_count,
        }
        if self.estimated_cost is not None:
            payload["estimated_cost"] = self.estimated_cost
        if self.error_type:
            payload["error_type"] = self.error_type
        if self.status_code is not None:
            payload["status_code"] = self.status_code
        return payload


class LlmClient:
    def __init__(self, component: str = "LlmClient") -> None:
        self.settings = get_settings()
        self.last_failure: LlmCallFailure | None = None
        self.component = component
        self.last_call: ModelCallTelemetry | None = None
        self.call_history: list[ModelCallTelemetry] = []

    def is_configured(self) -> bool:
        return bool(self.settings.llm_api_key and self.settings.llm_base_url and self.settings.llm_model)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | None = None,
        operation: str = "chat_completion",
    ) -> str | None:
        self.last_failure = None
        self.last_call = None
        started = time.perf_counter()
        if not self.is_configured():
            self.last_failure = LlmCallFailure(
                error_type="not_configured",
                message="LLM API key/base URL/model is missing",
                attempt=0,
                max_attempts=0,
            )
            self._record_telemetry(
                operation=operation,
                stream=False,
                status="failed",
                started=started,
                attempt_count=0,
                error_type="not_configured",
            )
            return None
        url = f"{self.settings.llm_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "stream": False,
        }
        thinking_type = self._deepseek_thinking_type()
        if thinking_type:
            payload["thinking"] = {"type": thinking_type}
        if response_format is not None:
            payload["response_format"] = response_format

        max_retries = max(0, int(getattr(self.settings, "llm_max_retries", 2)))
        max_attempts = max_retries + 1
        backoff_seconds = max(0.0, float(getattr(self.settings, "llm_retry_backoff_seconds", 0.8)))
        timeout_seconds = float(getattr(self.settings, "llm_timeout_seconds", 60))

        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            for attempt in range(1, max_attempts + 1):
                try:
                    response = await client.post(url, headers=headers, json=payload)
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    failure = LlmCallFailure(
                        error_type=type(exc).__name__,
                        message=str(exc),
                        attempt=attempt,
                        max_attempts=max_attempts,
                    )
                    retrying = attempt < max_attempts
                    self._record_failure(failure, retrying=retrying)
                    if retrying:
                        await self._sleep_before_retry(backoff_seconds, attempt)
                        continue
                    self._record_telemetry(
                        operation=operation,
                        stream=False,
                        status="failed",
                        started=started,
                        attempt_count=attempt,
                        error_type=failure.error_type,
                    )
                    return None
                except Exception as exc:
                    failure = LlmCallFailure(
                        error_type=type(exc).__name__,
                        message=str(exc),
                        attempt=attempt,
                        max_attempts=max_attempts,
                    )
                    self._record_failure(failure, retrying=False)
                    self._record_telemetry(
                        operation=operation,
                        stream=False,
                        status="failed",
                        started=started,
                        attempt_count=attempt,
                        error_type=failure.error_type,
                    )
                    return None

                status_code = int(getattr(response, "status_code", 200))
                # 400 + response_format 不支持 → 立刻去掉 response_format 重试
                if status_code == 400 and response_format is not None and "response_format" in payload:
                    try:
                        body_preview = response.text[:300]
                    except Exception:
                        body_preview = ""
                    if "response_format" in body_preview or "json_object" in body_preview:
                        retry_payload = dict(payload)
                        retry_payload.pop("response_format", None)
                        try:
                            response = await client.post(url, headers=headers, json=retry_payload)
                            status_code = int(getattr(response, "status_code", 200))
                            payload = retry_payload
                        except Exception:
                            pass
                if status_code >= 400:
                    failure = self._failure_from_response(response, attempt, max_attempts)
                    retrying = status_code in RETRYABLE_STATUS_CODES and attempt < max_attempts
                    self._record_failure(failure, retrying=retrying)
                    if retrying:
                        await self._sleep_before_retry(backoff_seconds, attempt)
                        continue
                    self._record_telemetry(
                        operation=operation,
                        stream=False,
                        status="failed",
                        started=started,
                        attempt_count=attempt,
                        error_type=failure.error_type,
                        status_code=failure.status_code,
                    )
                    return None

                try:
                    data = response.json()
                    self.last_failure = None
                    self._record_telemetry(
                        operation=operation,
                        stream=False,
                        status="succeeded",
                        started=started,
                        attempt_count=attempt,
                        usage=self._usage_from_response(data),
                    )
                    content = str(data["choices"][0]["message"]["content"])
                    _log_prompt(self.component, operation, system_prompt, user_prompt,
                                content, duration_ms=(time.perf_counter() - started) * 1000)
                    return content
                except Exception as exc:
                    failure = LlmCallFailure(
                        error_type=type(exc).__name__,
                        message=f"LLM response parse failed: {exc}",
                        attempt=attempt,
                        max_attempts=max_attempts,
                    )
                    self._record_failure(failure, retrying=False)
                    self._record_telemetry(
                        operation=operation,
                        stream=False,
                        status="failed",
                        started=started,
                        attempt_count=attempt,
                        error_type=failure.error_type,
                    )
                    return None
        return None

    async def generate_required(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | None = None,
        operation: str = "chat_completion",
    ) -> str:
        content = await self.generate(system_prompt, user_prompt, response_format=response_format, operation=operation)
        if content is None:
            detail = self.last_failure.summary() if self.last_failure is not None else "unknown failure"
            raise RuntimeError(f"LLM API 未配置或调用失败：{detail}")
        return content

    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | None = None,
        operation: str = "chat_completion_stream",
    ) -> AsyncGenerator[str, None]:
        self.last_failure = None
        self.last_call = None
        started = time.perf_counter()
        if not self.is_configured():
            self.last_failure = LlmCallFailure(
                error_type="not_configured",
                message="LLM API key/base URL/model is missing",
                attempt=0,
                max_attempts=0,
            )
            self._record_telemetry(
                operation=operation,
                stream=True,
                status="failed",
                started=started,
                attempt_count=0,
                error_type="not_configured",
            )
            return

        url = f"{self.settings.llm_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "stream": True,
        }
        thinking_type = self._deepseek_thinking_type()
        if thinking_type:
            payload["thinking"] = {"type": thinking_type}
        if bool(getattr(self.settings, "llm_stream_include_usage", True)):
            payload["stream_options"] = {"include_usage": True}
        if response_format is not None:
            payload["response_format"] = response_format

        max_retries = max(0, int(getattr(self.settings, "llm_max_retries", 2)))
        max_attempts = max_retries + 1
        backoff_seconds = max(0.0, float(getattr(self.settings, "llm_retry_backoff_seconds", 0.8)))
        timeout_seconds = float(getattr(self.settings, "llm_timeout_seconds", 60))

        _stream_buffer: list[str] = []
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            for attempt in range(1, max_attempts + 1):
                usage: dict[str, Any] = {}
                try:
                    async with client.stream("POST", url, headers=headers, json=payload) as response:
                        status_code = int(getattr(response, "status_code", 200))
                        if status_code >= 400:
                            body = (await response.aread()).decode("utf-8", errors="replace")[:500]
                            if (
                                status_code == 400
                                and "stream_options" in payload
                                and ("stream_options" in body or "include_usage" in body)
                            ):
                                payload.pop("stream_options", None)
                                async with client.stream("POST", url, headers=headers, json=payload) as retry_response:
                                    retry_status = int(getattr(retry_response, "status_code", 200))
                                    if retry_status < 400:
                                        async for line in retry_response.aiter_lines():
                                            data = self._stream_data_from_line(line)
                                            if data == "[DONE]":
                                                self.last_failure = None
                                                self._record_telemetry(
                                                    operation=operation,
                                                    stream=True,
                                                    status="succeeded",
                                                    started=started,
                                                    attempt_count=attempt,
                                                    usage=usage,
                                                )
                                                return
                                            if not isinstance(data, dict):
                                                continue
                                            usage = self._usage_from_response(data) or usage
                                            delta = self._stream_delta_from_data(data)
                                            if delta:
                                                yield delta
                                        self.last_failure = None
                                        self._record_telemetry(
                                            operation=operation,
                                            stream=True,
                                            status="succeeded",
                                            started=started,
                                            attempt_count=attempt,
                                            usage=usage,
                                        )
                                        return
                                    body = (await retry_response.aread()).decode("utf-8", errors="replace")[:500]
                                    status_code = retry_status
                            failure = LlmCallFailure(
                                error_type="http_status_error",
                                message=f"LLM API returned HTTP {status_code}",
                                attempt=attempt,
                                max_attempts=max_attempts,
                                status_code=status_code,
                                response_body_preview=body,
                            )
                            retrying = status_code in RETRYABLE_STATUS_CODES and attempt < max_attempts
                            self._record_failure(failure, retrying=retrying)
                            if retrying:
                                await self._sleep_before_retry(backoff_seconds, attempt)
                                continue
                            self._record_telemetry(
                                operation=operation,
                                stream=True,
                                status="failed",
                                started=started,
                                attempt_count=attempt,
                                error_type=failure.error_type,
                                status_code=failure.status_code,
                            )
                            return

                        async for line in response.aiter_lines():
                            data = self._stream_data_from_line(line)
                            if data is None:
                                continue
                            if data == "[DONE]":
                                self.last_failure = None
                                self._record_telemetry(
                                    operation=operation,
                                    stream=True,
                                    status="succeeded",
                                    started=started,
                                    attempt_count=attempt,
                                    usage=usage,
                                )
                                _log_prompt(self.component, operation, system_prompt, user_prompt,
                                            "".join(_stream_buffer),
                                            duration_ms=(time.perf_counter() - started) * 1000)
                                return
                            if not isinstance(data, dict):
                                continue
                            usage = self._usage_from_response(data) or usage
                            delta = self._stream_delta_from_data(data)
                            if delta is None:
                                continue
                            _stream_buffer.append(delta)
                            yield delta
                        self.last_failure = None
                        self._record_telemetry(
                            operation=operation,
                            stream=True,
                            status="succeeded",
                            started=started,
                            attempt_count=attempt,
                            usage=usage,
                        )
                        _log_prompt(self.component, operation, system_prompt, user_prompt,
                                    "".join(_stream_buffer),
                                    duration_ms=(time.perf_counter() - started) * 1000)
                        return
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    failure = LlmCallFailure(
                        error_type=type(exc).__name__,
                        message=str(exc),
                        attempt=attempt,
                        max_attempts=max_attempts,
                    )
                    retrying = attempt < max_attempts
                    self._record_failure(failure, retrying=retrying)
                    if retrying:
                        await self._sleep_before_retry(backoff_seconds, attempt)
                        continue
                    self._record_telemetry(
                        operation=operation,
                        stream=True,
                        status="failed",
                        started=started,
                        attempt_count=attempt,
                        error_type=failure.error_type,
                    )
                    return
                except Exception as exc:
                    failure = LlmCallFailure(
                        error_type=type(exc).__name__,
                        message=str(exc),
                        attempt=attempt,
                        max_attempts=max_attempts,
                    )
                    self._record_failure(failure, retrying=False)
                    self._record_telemetry(
                        operation=operation,
                        stream=True,
                        status="failed",
                        started=started,
                        attempt_count=attempt,
                        error_type=failure.error_type,
                    )
                    return

    async def generate_stream_required(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | None = None,
        operation: str = "chat_completion_stream",
    ) -> AsyncGenerator[str, None]:
        saw_content = False
        async for delta in self.generate_stream(
            system_prompt,
            user_prompt,
            response_format=response_format,
            operation=operation,
        ):
            saw_content = True
            yield delta
        if not saw_content and self.last_failure is not None:
            raise RuntimeError(f"LLM API 未配置或调用失败：{self.last_failure.summary()}")

    def _failure_from_response(
        self,
        response: httpx.Response,
        attempt: int,
        max_attempts: int,
    ) -> LlmCallFailure:
        status_code = int(getattr(response, "status_code", 0) or 0)
        body = str(getattr(response, "text", "") or "")[:500]
        return LlmCallFailure(
            error_type="http_status_error",
            message=f"LLM API returned HTTP {status_code}",
            attempt=attempt,
            max_attempts=max_attempts,
            status_code=status_code,
            response_body_preview=body,
        )

    def _record_failure(self, failure: LlmCallFailure, *, retrying: bool) -> None:
        self.last_failure = failure
        logger.warning(
            "LLM API call failed: attempt=%s/%s error_type=%s status_code=%s retrying=%s response_body_preview=%s",
            failure.attempt,
            failure.max_attempts,
            failure.error_type,
            failure.status_code,
            retrying,
            failure.response_body_preview,
        )

    def _stream_data_from_line(self, line: str) -> dict[str, Any] | str | None:
        if not line or line.startswith(":") or not line.startswith("data:"):
            return None
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            return "[DONE]"
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    def _stream_delta_from_line(self, line: str) -> str | None:
        data = self._stream_data_from_line(line)
        if data == "[DONE]":
            return "[DONE]"
        if not isinstance(data, dict):
            return None
        return self._stream_delta_from_data(data)

    def _stream_delta_from_data(self, data: dict[str, Any]) -> str | None:
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            return None
        first = choices[0] if isinstance(choices[0], dict) else {}
        delta = first.get("delta") if isinstance(first.get("delta"), dict) else {}
        content = delta.get("content")
        return str(content) if content else None

    def telemetry_payloads(self) -> list[dict[str, Any]]:
        return [item.model_dump() for item in self.call_history]

    def _record_telemetry(
        self,
        *,
        operation: str,
        stream: bool,
        status: str,
        started: float,
        attempt_count: int,
        usage: dict[str, Any] | None = None,
        error_type: str | None = None,
        status_code: int | None = None,
    ) -> None:
        telemetry = ModelCallTelemetry(
            component=self.component,
            model=str(self.settings.llm_model or ""),
            provider=self._provider_from_base_url(),
            operation=operation,
            stream=stream,
            status=status,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            usage=usage or {},
            estimated_cost=self._estimate_cost(usage or {}),
            attempt_count=attempt_count,
            error_type=error_type,
            status_code=status_code,
        )
        self.last_call = telemetry
        self.call_history.append(telemetry)
        logger.info("LLM telemetry: %s", telemetry.model_dump())

    def _usage_from_response(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict) or not isinstance(data.get("usage"), dict):
            return {}
        return json.loads(json.dumps(data["usage"], ensure_ascii=False, default=str))

    def _estimate_cost(self, usage: dict[str, Any]) -> float | None:
        input_price = getattr(self.settings, "llm_input_price_per_1k", None)
        output_price = getattr(self.settings, "llm_output_price_per_1k", None)
        if input_price is None and output_price is None:
            return None
        input_tokens = self._token_count(usage, "prompt_tokens", "input_tokens")
        output_tokens = self._token_count(usage, "completion_tokens", "output_tokens")
        cost = 0.0
        if input_price is not None:
            cost += input_tokens / 1000.0 * float(input_price)
        if output_price is not None:
            cost += output_tokens / 1000.0 * float(output_price)
        return round(cost, 8)

    def _token_count(self, usage: dict[str, Any], *keys: str) -> int:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                return int(value)
        return 0

    def _provider_from_base_url(self) -> str:
        host = urlparse(str(self.settings.llm_base_url or "")).netloc
        return host or "local"

    def _deepseek_thinking_type(self) -> str:
        value = str(getattr(self.settings, "llm_thinking_type", "") or "").strip().lower()
        if value not in {"enabled", "disabled"}:
            return ""
        provider = self._provider_from_base_url().lower()
        model = str(getattr(self.settings, "llm_model", "") or "").lower()
        if "deepseek" not in provider and "deepseek" not in model:
            return ""
        return value

    async def _sleep_before_retry(self, backoff_seconds: float, attempt: int) -> None:
        if backoff_seconds <= 0:
            return
        await asyncio.sleep(backoff_seconds * (2 ** (attempt - 1)))
