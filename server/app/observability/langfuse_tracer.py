from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class LangfuseTraceRun:
    """Best-effort Langfuse trace context.

    The RAG pipeline must never fail because observability fails, so every SDK
    call is guarded and degrades to a no-op.
    """

    def __init__(
        self,
        tracer: "LangfuseTracer",
        name: str,
        input_payload: dict[str, Any] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.tracer = tracer
        self.name = name
        self.input_payload = tracer._safe_payload(input_payload or {})
        self.metadata = tracer._safe_payload(metadata or {})
        self.user_id = user_id
        self.session_id = session_id
        self._root_cm: Any | None = None
        self._root: Any | None = None
        self._propagation_cm: Any | None = None
        self._ended = False
        self._start()

    @property
    def enabled(self) -> bool:
        return bool(self.tracer.sdk_enabled and self._root is not None)

    async def span(
        self,
        name: str,
        *,
        input_payload: dict[str, Any] | None = None,
        output_payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        as_type: str = "span",
    ) -> None:
        if not self.enabled:
            return
        try:
            self._record_span(
                name,
                input_payload=input_payload,
                output_payload=output_payload,
                metadata=metadata,
                as_type=as_type,
            )
        except Exception as exc:
            logger.warning("Langfuse span failed for %s: %s", name, exc)

    async def end(
        self,
        *,
        output_payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if self._ended:
            return
        self._ended = True
        if not self.enabled:
            return
        try:
            output = self.tracer._safe_payload(output_payload or {})
            update_payload: dict[str, Any] = {}
            if output:
                update_payload["output"] = output
            if metadata or error:
                update_payload["metadata"] = self.tracer._safe_payload(
                    {
                        **(metadata or {}),
                        **({"error": error} if error else {}),
                    }
                )
            if update_payload and hasattr(self._root, "update"):
                self._root.update(**update_payload)
            if output and hasattr(self._root, "set_trace_io"):
                self._root.set_trace_io(input=self.input_payload, output=output)
        except Exception as exc:
            logger.warning("Langfuse trace end update failed for %s: %s", self.name, exc)
        finally:
            self._close()
            self.tracer.flush()

    def _start(self) -> None:
        if not self.tracer.sdk_enabled:
            return
        try:
            kwargs = {
                "as_type": "span",
                "name": self.name,
                "input": self.input_payload,
                "metadata": self.metadata,
            }
            self._root_cm = self.tracer._start_observation(**kwargs)
            self._root = self._root_cm.__enter__()
            self._enter_propagation()
        except TypeError:
            try:
                self._root_cm = self.tracer._start_observation(as_type="span", name=self.name)
                self._root = self._root_cm.__enter__()
                if hasattr(self._root, "update"):
                    self._root.update(input=self.input_payload, metadata=self.metadata)
                self._enter_propagation()
            except Exception as exc:
                logger.warning("Langfuse trace start failed for %s: %s", self.name, exc)
                self._root = None
                self._root_cm = None
        except Exception as exc:
            logger.warning("Langfuse trace start failed for %s: %s", self.name, exc)
            self._root = None
            self._root_cm = None

    def _enter_propagation(self) -> None:
        propagate = self.tracer.propagate_attributes
        if not propagate:
            return
        try:
            metadata = {"pipeline": "ecommerce_rag"}
            kwargs: dict[str, Any] = {
                "trace_name": self.name,
                "metadata": metadata,
            }
            if self.user_id:
                kwargs["user_id"] = str(self.user_id)[:200]
            if self.session_id:
                kwargs["session_id"] = str(self.session_id)[:200]
            self._propagation_cm = propagate(**kwargs)
            self._propagation_cm.__enter__()
        except TypeError:
            try:
                self._propagation_cm = propagate(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    metadata={"pipeline": "ecommerce_rag"},
                )
                self._propagation_cm.__enter__()
            except Exception as exc:
                logger.warning("Langfuse propagation setup failed for %s: %s", self.name, exc)
                self._propagation_cm = None
        except Exception as exc:
            logger.warning("Langfuse propagation setup failed for %s: %s", self.name, exc)
            self._propagation_cm = None

    def _record_span(
        self,
        name: str,
        *,
        input_payload: dict[str, Any] | None,
        output_payload: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
        as_type: str,
    ) -> None:
        input_payload = self.tracer._safe_payload(input_payload or {})
        output_payload = self.tracer._safe_payload(output_payload or {})
        metadata = self.tracer._safe_payload(metadata or {})
        kwargs: dict[str, Any] = {"as_type": as_type, "name": name}
        if input_payload:
            kwargs["input"] = input_payload
        if metadata:
            kwargs["metadata"] = metadata

        try:
            cm = self.tracer._start_observation(parent=self._root, **kwargs)
        except TypeError:
            cm = self.tracer._start_observation(parent=self._root, as_type=as_type, name=name)

        with cm as observation:
            update_payload: dict[str, Any] = {}
            if input_payload:
                update_payload["input"] = input_payload
            if output_payload:
                update_payload["output"] = output_payload
            if metadata:
                update_payload["metadata"] = metadata
            if update_payload and hasattr(observation, "update"):
                observation.update(**update_payload)

    def _close(self) -> None:
        if self._propagation_cm is not None:
            try:
                self._propagation_cm.__exit__(None, None, None)
            except Exception as exc:
                logger.warning("Langfuse propagation close failed for %s: %s", self.name, exc)
            self._propagation_cm = None
        if self._root_cm is not None:
            try:
                self._root_cm.__exit__(None, None, None)
            except Exception as exc:
                logger.warning("Langfuse root span close failed for %s: %s", self.name, exc)
            self._root_cm = None
            self._root = None


class LangfuseTracer:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.enabled = bool(
            self.settings.langfuse_public_key
            and self.settings.langfuse_secret_key
            and self.settings.langfuse_host
        )
        self.client: Any | None = None
        self.propagate_attributes: Any | None = None
        self.sdk_enabled = False
        self._configure_sdk()

    def start_run(
        self,
        name: str,
        *,
        input_payload: dict[str, Any] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LangfuseTraceRun:
        return LangfuseTraceRun(
            self,
            name,
            input_payload=input_payload,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
        )

    async def trace(self, name: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        safe_payload = self._safe_payload(payload)
        if self.sdk_enabled:
            try:
                with self._start_observation(as_type="span", name=name, metadata=safe_payload):
                    pass
                self.flush()
                return
            except Exception as exc:
                logger.warning("Langfuse SDK event failed for %s: %s", name, exc)
        try:
            await self._post_event(name, safe_payload)
        except Exception as exc:
            logger.warning("Langfuse ingestion event failed for %s: %s", name, exc)

    def flush(self) -> None:
        if not self.client or not hasattr(self.client, "flush"):
            return
        try:
            self.client.flush()
        except Exception as exc:
            logger.warning("Langfuse flush failed: %s", exc)

    def _configure_sdk(self) -> None:
        if not self.enabled:
            return
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", self.settings.langfuse_public_key or "")
        os.environ.setdefault("LANGFUSE_SECRET_KEY", self.settings.langfuse_secret_key or "")
        os.environ.setdefault("LANGFUSE_HOST", self.settings.langfuse_host or "")
        try:
            from langfuse import get_client, propagate_attributes

            self.client = get_client()
            self.propagate_attributes = propagate_attributes
            self.sdk_enabled = bool(hasattr(self.client, "start_as_current_observation"))
        except Exception as exc:
            logger.warning("Langfuse SDK unavailable, using ingestion event fallback only: %s", exc)
            self.client = None
            self.propagate_attributes = None
            self.sdk_enabled = False

    def _start_observation(self, parent: Any | None = None, **kwargs: Any) -> Any:
        if parent is not None and hasattr(parent, "start_as_current_observation"):
            return parent.start_as_current_observation(**kwargs)
        if not self.client:
            raise RuntimeError("Langfuse SDK client is not configured")
        return self.client.start_as_current_observation(**kwargs)

    async def _post_event(self, name: str, payload: dict[str, Any]) -> None:
        url = f"{self.settings.langfuse_host.rstrip('/')}/api/public/ingestion"
        body = {
            "batch": [
                {
                    "type": "event-create",
                    "body": {
                        "name": name,
                        "metadata": payload,
                    },
                }
            ]
        }
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.post(
                url,
                auth=(self.settings.langfuse_public_key or "", self.settings.langfuse_secret_key or ""),
                json=body,
            )
            response.raise_for_status()

    def _safe_payload(self, payload: dict[str, Any] | list[Any] | Any) -> Any:
        def simplify(value: Any) -> Any:
            if isinstance(value, dict):
                if {"product_id", "name"} & set(value):
                    return {
                        key: value.get(key)
                        for key in [
                            "product_id",
                            "name",
                            "category",
                            "sub_category",
                            "price",
                            "rerank_score",
                            "coverage_reason",
                        ]
                        if key in value
                    }
                return {str(key): simplify(item) for key, item in value.items()}
            if isinstance(value, list):
                return [simplify(item) for item in value[:30]]
            return value

        return simplify(payload)
