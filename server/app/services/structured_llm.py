from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable
from typing import Any


JsonValidator = Callable[[dict[str, Any]], list[str]]


class StructuredLlmValidationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        errors: list[str],
        data: dict[str, Any] | None,
        content: str,
    ) -> None:
        super().__init__(message)
        self.errors = errors
        self.data = data
        self.content = content


async def generate_validated_json(
    llm_client: Any,
    system_prompt: str,
    user_prompt: str,
    *,
    validate: JsonValidator,
    error_message: str,
    max_retries: int = 1,
    response_format: dict[str, Any] | None = None,
    operation: str = "structured_json",
) -> dict[str, Any]:
    current_user_prompt = user_prompt
    last_errors: list[str] = []
    last_data: dict[str, Any] | None = None
    last_content = ""

    for attempt in range(max_retries + 1):
        content = await _generate_required(
            llm_client,
            system_prompt,
            current_user_prompt,
            response_format=response_format,
            operation=operation if attempt == 0 else f"{operation}.repair",
        )
        last_content = content
        data = parse_json_object(content)
        last_data = data
        if data is None:
            last_errors = ["输出不是可解析的 JSON object。"]
        else:
            last_errors = validate(data)
            if not last_errors:
                return data
        if attempt < max_retries:
            current_user_prompt = _repair_prompt(user_prompt, last_content, last_errors)

    raise StructuredLlmValidationError(
        error_message,
        errors=last_errors,
        data=last_data,
        content=last_content,
    )


def parse_json_object(content: str) -> dict[str, Any] | None:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def _generate_required(
    llm_client: Any,
    system_prompt: str,
    user_prompt: str,
    *,
    response_format: dict[str, Any] | None,
    operation: str,
) -> str:
    generate_required = llm_client.generate_required
    kwargs: dict[str, Any] = {}
    if response_format is not None and _supports_parameter(generate_required, "response_format"):
        kwargs["response_format"] = response_format
    if _supports_parameter(generate_required, "operation"):
        kwargs["operation"] = operation
    return await generate_required(system_prompt, user_prompt, **kwargs)


def _supports_parameter(callable_obj: Any, name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _repair_prompt(original_user_prompt: str, previous_output: str, errors: list[str]) -> str:
    return (
        f"{original_user_prompt}\n\n"
        "上一次输出的 JSON 结构不符合要求。请根据以下错误修正，并只返回一个 JSON object，"
        "不要输出 Markdown 或解释文字。\n"
        "结构错误：\n"
        + "\n".join(f"- {error}" for error in errors)
        + "\n\n上一次输出：\n"
        + previous_output[:4000]
    )
