from __future__ import annotations

import base64
import hashlib
import io
import math
import mimetypes
from http import HTTPStatus
from pathlib import Path
from threading import Lock
from typing import Any

import httpx

from app.core.config import get_settings


class EmbeddingClient:
    _clip_lock = Lock()
    _clip_model = None
    _clip_preprocess = None
    _clip_device = "cpu"

    def __init__(self) -> None:
        self.settings = get_settings()

    def embed(self, text: str) -> list[float]:
        if not self._is_configured():
            return self._hash_embedding(text)
        for attempt in range(2):
            try:
                return self._remote_embedding(text)
            except Exception:
                if attempt == 0:
                    continue
        return self._hash_embedding(text)

    def embed_image(self, image_path: str | Path) -> list[float]:
        if self._image_remote_is_configured():
            for attempt in range(2):
                try:
                    return self._remote_image_embedding(image_path)
                except Exception:
                    if attempt == 0:
                        continue
        try:
            return self._local_clip_image_embedding(image_path)
        except Exception:
            return self._hash_file_embedding(image_path)

    def embed_joint(self, text: str, image_path: str | Path | None = None) -> list[float]:
        if image_path is None:
            return self.embed(text)
        image_embedding = self.embed_image(image_path)
        if not text.strip():
            return image_embedding
        text_embedding = self.embed(text)
        values = [
            (image_value + text_value) / 2.0
            for image_value, text_value in zip(image_embedding, text_embedding, strict=False)
        ]
        return self._normalize(values, dim=int(self.settings.image_embedding_dim))

    def _is_configured(self) -> bool:
        return bool(
            self.settings.embedding_api_key
            and self.settings.embedding_base_url
            and self.settings.embedding_model
        )

    def _image_remote_is_configured(self) -> bool:
        backend = str(self.settings.image_embedding_backend or "auto").lower()
        if backend in {"local", "clip", "cn_clip", "hash", "none", "off"}:
            return False
        return bool(self._image_embedding_api_key() and self.settings.image_embedding_model)

    def _image_embedding_api_key(self) -> str | None:
        return self.settings.image_embedding_api_key or self.settings.dashscope_api_key

    def _remote_embedding(self, text: str) -> list[float]:
        url = f"{self.settings.embedding_base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.settings.embedding_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.embedding_model,
            "input": text,
        }
        with httpx.Client(timeout=self.settings.embedding_timeout_seconds) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        embedding = data["data"][0]["embedding"]
        return [float(value) for value in embedding]

    def _remote_image_embedding(self, image_path: str | Path) -> list[float]:
        import dashscope

        response = dashscope.MultiModalEmbedding.call(
            api_key=self._image_embedding_api_key(),
            model=self.settings.image_embedding_model,
            input=[{"image": self._image_to_data_uri(image_path)}],
        )
        status_code = self._response_field(response, "status_code")
        if status_code is not None and int(status_code) != HTTPStatus.OK:
            code = self._response_field(response, "code", "")
            message = self._response_field(response, "message", "")
            raise RuntimeError(f"DashScope image embedding failed: {status_code} {code} {message}".strip())

        output = self._response_field(response, "output") or {}
        embeddings = output.get("embeddings") if isinstance(output, dict) else None
        if not embeddings:
            raise RuntimeError("DashScope image embedding response missing output.embeddings")

        embedding = embeddings[0].get("embedding") if isinstance(embeddings[0], dict) else None
        if not embedding:
            raise RuntimeError("DashScope image embedding response missing embedding vector")
        return self._normalize([float(value) for value in embedding], dim=int(self.settings.image_embedding_dim))

    def _image_to_data_uri(self, image_path: str | Path) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            from PIL import Image

            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((640, 640))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=80, optimize=True)
            data = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{data}"
        except Exception:
            pass
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        if mime_type == "image/jpg":
            mime_type = "image/jpeg"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{data}"

    def _response_field(self, response: Any, name: str, default: Any = None) -> Any:
        if isinstance(response, dict):
            return response.get(name, default)
        return getattr(response, name, default)

    def _hash_embedding(self, text: str) -> list[float]:
        return self._hash_embedding_with_dim(text, int(self.settings.embedding_dim))

    def _hash_embedding_with_dim(self, text: str, dim: int) -> list[float]:
        dim = max(1, dim)
        values = [0.0] * dim
        tokens = text.lower().split()
        if not tokens:
            tokens = [text.lower()]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8", errors="ignore")).digest()
            for offset in range(0, len(digest), 4):
                index = int.from_bytes(digest[offset : offset + 2], "big") % dim
                sign = 1.0 if digest[offset + 2] % 2 == 0 else -1.0
                values[index] += sign
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    def _hash_file_embedding(self, image_path: str | Path) -> list[float]:
        path = Path(image_path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else str(path)
        return self._hash_embedding_with_dim(digest, int(self.settings.image_embedding_dim))

    def _normalize(self, values: list[float], *, dim: int | None = None) -> list[float]:
        target_dim = max(1, int(dim or self.settings.embedding_dim))
        if len(values) < target_dim:
            values = values + [0.0] * (target_dim - len(values))
        values = values[:target_dim]
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    def _local_clip_image_embedding(self, image_path: str | Path) -> list[float]:
        import torch
        from PIL import Image
        from cn_clip.clip import load_from_name

        with self._clip_lock:
            if self.__class__._clip_model is None or self.__class__._clip_preprocess is None:
                device = self._resolve_clip_device(torch)
                model, preprocess = load_from_name(self.settings.clip_model_name, device=device, download_root=None)
                model.eval()
                self.__class__._clip_model = model
                self.__class__._clip_preprocess = preprocess
                self.__class__._clip_device = device

        model = self.__class__._clip_model
        preprocess = self.__class__._clip_preprocess
        device = self.__class__._clip_device
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(image)
            features = features / features.norm(dim=-1, keepdim=True)
        vector = features[0].detach().cpu().tolist()
        return self._normalize([float(value) for value in vector], dim=int(self.settings.image_embedding_dim))

    def _resolve_clip_device(self, torch_module) -> str:
        configured = str(self.settings.clip_device or "auto").lower()
        if configured != "auto":
            return configured
        return "cuda" if torch_module.cuda.is_available() else "cpu"
