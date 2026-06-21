from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.schemas import ChatStreamRequest
from app.services.upload_storage import resolve_image_path


@dataclass(frozen=True)
class NormalizedInput:
    text: str
    image_path: Path | None = None

    def __str__(self) -> str:
        return self.text

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.text == other
        if isinstance(other, NormalizedInput):
            return self.text == other.text and self.image_path == other.image_path
        return NotImplemented


class InputProcessor:
    def normalize(self, request: ChatStreamRequest) -> NormalizedInput:
        text = " ".join(request.message.strip().split())
        image_path = resolve_image_path(request.image_id) if request.image_id else None
        return NormalizedInput(text=text, image_path=image_path)
