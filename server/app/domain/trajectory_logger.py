from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class TrajectoryLogger:
    def __init__(self) -> None:
        self.settings = get_settings()

    def log(self, payload: dict[str, Any]) -> None:
        if not self.settings.enable_trajectory_log:
            return
        path = Path(self.settings.trajectory_log_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "logged_at": datetime.now(timezone.utc).isoformat(),
                **payload,
            }
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to write multi_need trajectory log: %s", exc)
