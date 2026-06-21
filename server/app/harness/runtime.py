from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings, get_settings
from app.harness.budget_manager import BudgetManager
from app.harness.evidence_cache import EvidenceCache, InMemoryEvidenceCache
from app.harness.tool_registry import ToolRegistry
from app.harness.trace_recorder import TraceRecorder

@dataclass
class HarnessRuntime:
    """Harness 基础设施运行时（HarnessRuntime）。

    聚合预算、trace、工具注册和短期 evidence cache。它不裁决 route，也不执行 worker。
    """

    budget_manager: BudgetManager
    trace_recorder: TraceRecorder
    tool_registry: ToolRegistry
    evidence_cache: EvidenceCache

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "HarnessRuntime":
        resolved = settings or get_settings()
        return cls(
            budget_manager=BudgetManager(),
            trace_recorder=TraceRecorder(),
            tool_registry=ToolRegistry(),
            evidence_cache=cls._build_evidence_cache(resolved),
        )

    @classmethod
    def _build_evidence_cache(cls, settings: Settings) -> EvidenceCache:
        return InMemoryEvidenceCache(
            ttl_seconds=settings.evidence_cache_ttl_seconds,
            recent_turns=settings.evidence_cache_recent_turns,
            max_candidates_per_turn=settings.evidence_cache_max_candidates_per_turn,
        )
