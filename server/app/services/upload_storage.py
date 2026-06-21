from __future__ import annotations

import secrets
from datetime import datetime, timedelta
from pathlib import Path


_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_MAX_IMAGE_BYTES = 10 * 1024 * 1024


def get_upload_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    upload_dir = repo_root / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def save_uploaded_image(filename: str, content: bytes) -> tuple[str, Path]:
    if len(content) > _MAX_IMAGE_BYTES:
        raise ValueError(f"图片过大，最大支持 {_MAX_IMAGE_BYTES // 1024 // 1024}MB")

    ext = Path(filename or "").suffix.lower() or ".jpg"
    if ext not in _ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(_ALLOWED_EXTENSIONS))
        raise ValueError(f"不支持的图片格式 {ext}，仅支持 {allowed}")

    image_id = secrets.token_urlsafe(16)
    target = get_upload_dir() / f"{image_id}{ext}"
    target.write_bytes(content)
    return image_id, target


def resolve_image_path(image_id: str | None) -> Path | None:
    if not image_id:
        return None
    upload_dir = get_upload_dir()
    for ext in _ALLOWED_EXTENSIONS:
        path = upload_dir / f"{image_id}{ext}"
        if path.exists():
            return path
    return None


def cleanup_expired(ttl_hours: int = 24) -> int:
    upload_dir = get_upload_dir()
    cutoff = datetime.now() - timedelta(hours=ttl_hours)
    deleted = 0
    for path in upload_dir.iterdir():
        if path.is_file() and datetime.fromtimestamp(path.stat().st_mtime) < cutoff:
            path.unlink(missing_ok=True)
            deleted += 1
    return deleted
