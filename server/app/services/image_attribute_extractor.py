from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import mimetypes
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from app.core.config import get_settings
from app.schemas import ImageAttributes
from app.services.llm_client import ModelCallTelemetry
from app.services.structured_llm import parse_json_object


logger = logging.getLogger(__name__)


class ImageAttributeExtractor:
    """图片属性理解服务（ImageAttributeExtractor）。

    这是 IntentPlanner 前置输入增强 Service：只把本轮图片转成结构化视觉语义推测，
    不检索商品、不决定 route，也不作为商品事实源。
    """

    RESPONSE_FORMAT = {"type": "json_object"}

    def __init__(self) -> None:
        self.settings = get_settings()
        self.last_call: ModelCallTelemetry | None = None
        self.call_history: list[ModelCallTelemetry] = []

    async def extract(self, image_path: str | Path, user_text: str = "") -> ImageAttributes:
        if not self._is_configured():
            return ImageAttributes(
                available=False,
                uncertainty_note="VLM is not configured.",
            )
        path = Path(image_path)
        if not path.exists():
            return ImageAttributes(
                available=False,
                uncertainty_note="Image file is not available.",
            )

        attempts = max(1, int(getattr(self.settings, "vlm_max_retries", 1)) + 1)
        last_error = ""
        for attempt in range(1, attempts + 1):
            try:
                deadline_seconds = max(1.0, float(getattr(self.settings, "vlm_timeout_seconds", 8)))
                started = time.perf_counter()
                vlm_task = asyncio.create_task(self._extract_once(path, user_text))
                try:
                    return await asyncio.wait_for(asyncio.shield(vlm_task), timeout=deadline_seconds)
                except TimeoutError:
                    self._record_telemetry(
                        started=started,
                        status="failed",
                        usage={},
                        error_type="TimeoutError",
                    )
                    vlm_task.add_done_callback(self._discard_background_task)
                    raise
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "ImageAttributeExtractor failed: attempt=%s/%s error=%s",
                    attempt,
                    attempts,
                    exc,
                )
                if attempt < attempts:
                    await asyncio.sleep(0.25 * attempt)
        return ImageAttributes(
            available=False,
            uncertainty_note=f"VLM extraction failed: {last_error[:160]}",
        )

    async def _extract_once(self, image_path: Path, user_text: str) -> ImageAttributes:
        started = time.perf_counter()
        usage: dict[str, Any] = {}
        url = f"{self.settings.vlm_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.vlm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.vlm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是电商导购系统的图片属性理解服务（ImageAttributeExtractor）。"
                        "你的职责是把用户上传图片转成结构化视觉语义推测，不推荐商品，不回答用户。"
                        "只输出 JSON object，不要输出 Markdown。\n\n"
                        "约束：\n"
                        "- 图片属性只是本轮推测，不是事实源。\n"
                        "- 用户文本只用于帮助判断商品目标；如果文本和图片冲突，保留不确定性说明。\n"
                        "- 不要复述用户原话，不要输出 original_query/vector_query/keyword_query/product_id。\n"
                        "- retrieval_query 只写适合图片相似召回的视觉商品词、颜色、风格、场景，最长 40 个中文字符。\n"
                        "- 不确定的材质、品类或场景必须写进 uncertainty_note。\n\n"
                        "返回 JSON schema: {"
                        "\"available\":true,"
                        "\"category_guess\":\"服饰鞋包\","
                        "\"product_type_guess\":\"外套\","
                        "\"colors\":[\"米白色\"],"
                        "\"style_tags\":[\"通勤\",\"简约\"],"
                        "\"material_guess\":\"针织或羊毛混纺\","
                        "\"occasion_tags\":[\"日常通勤\",\"春秋\"],"
                        "\"retrieval_query\":\"米白色 简约通勤 外套\","
                        "\"confidence\":0.78,"
                        "\"uncertainty_note\":\"材质仅根据图片推测\""
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "user_text": user_text,
                                    "task": "extract visual shopping attributes for planner context",
                                },
                                ensure_ascii=False,
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": self._image_to_data_uri(image_path)},
                        },
                    ],
                },
            ],
            "temperature": 0.1,
            "stream": False,
            "response_format": self.RESPONSE_FORMAT,
        }
        async with httpx.AsyncClient(timeout=float(self.settings.vlm_timeout_seconds)) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code == 400:
                    retry_payload = dict(payload)
                    retry_payload.pop("response_format", None)
                    response = await client.post(url, headers=headers, json=retry_payload)
                response.raise_for_status()
                data = response.json()
                usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
                self._record_telemetry(started=started, status="succeeded", usage=usage)
            except Exception as exc:
                status_code = None
                if "response" in locals():
                    status_code = int(getattr(response, "status_code", 0) or 0) or None
                self._record_telemetry(
                    started=started,
                    status="failed",
                    usage=usage,
                    error_type=type(exc).__name__,
                    status_code=status_code,
                )
                raise
        content = str(data["choices"][0]["message"]["content"])
        parsed = parse_json_object(content)
        if not isinstance(parsed, dict):
            raise RuntimeError("VLM response is not a JSON object.")
        return self._attributes_from_data(parsed)

    def _attributes_from_data(self, data: dict[str, Any]) -> ImageAttributes:
        confidence = self._float_or_zero(data.get("confidence"))
        confidence = max(0.0, min(1.0, confidence))
        return ImageAttributes(
            available=bool(data.get("available", True)),
            category_guess=self._short_text(data.get("category_guess"), limit=40),
            product_type_guess=self._short_text(data.get("product_type_guess"), limit=40),
            colors=self._string_list(data.get("colors"), limit=8),
            style_tags=self._string_list(data.get("style_tags"), limit=8),
            material_guess=self._short_text(data.get("material_guess"), limit=50),
            occasion_tags=self._string_list(data.get("occasion_tags"), limit=8),
            retrieval_query=self._short_text(data.get("retrieval_query"), limit=80),
            confidence=confidence,
            uncertainty_note=self._short_text(data.get("uncertainty_note"), limit=120),
        )

    def _image_to_data_uri(self, image_path: Path) -> str:
        try:
            from PIL import Image

            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image.thumbnail((512, 512))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=74, optimize=True)
            payload = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{payload}"
        except Exception:
            mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
            if mime_type == "image/jpg":
                mime_type = "image/jpeg"
            payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
            return f"data:{mime_type};base64,{payload}"

    def _is_configured(self) -> bool:
        return bool(self.settings.vlm_api_key and self.settings.vlm_base_url and self.settings.vlm_model)

    def telemetry_payloads(self) -> list[dict[str, Any]]:
        return [item.model_dump() for item in self.call_history]

    def _record_telemetry(
        self,
        *,
        started: float,
        status: str,
        usage: dict[str, Any],
        error_type: str | None = None,
        status_code: int | None = None,
    ) -> None:
        telemetry = ModelCallTelemetry(
            component="ImageAttributeExtractor",
            model=str(self.settings.vlm_model or ""),
            provider=urlparse(str(self.settings.vlm_base_url or "")).netloc or "vlm",
            operation="image_attribute_extractor.extract",
            stream=False,
            status=status,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            usage=json.loads(json.dumps(usage or {}, ensure_ascii=False, default=str)),
            estimated_cost=None,
            attempt_count=1,
            error_type=error_type,
            status_code=status_code,
        )
        self.last_call = telemetry
        self.call_history.append(telemetry)
        logger.info("VLM telemetry: %s", telemetry.model_dump())

    def _discard_background_task(self, task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.debug("Background VLM task finished with error: %s", exc)

    def _string_list(self, value: Any, *, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            text = self._short_text(item, limit=32)
            if text and text not in result:
                result.append(text)
            if len(result) >= limit:
                break
        return result

    def _short_text(self, value: Any, *, limit: int) -> str:
        text = " ".join(str(value or "").strip().split())
        return text[:limit]

    def _float_or_zero(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
