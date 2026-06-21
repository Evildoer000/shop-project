from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check benchmark report metrics for CI.")
    parser.add_argument("report", type=Path, help="Path to benchmark/report.json.")
    parser.add_argument("--min-total-turns", type=int, default=65)
    parser.add_argument("--min-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-route-ok", type=float, default=0.95)
    parser.add_argument("--min-forbidden-clean", type=float, default=0.95)
    parser.add_argument("--min-context-reuse-ok", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    overall = dict(report.get("overall") or {})
    failures: list[str] = []

    total_turns = int(report.get("total_turns") or overall.get("turns") or 0)
    if total_turns < args.min_total_turns:
        failures.append(f"total_turns {total_turns} < {args.min_total_turns}")

    _check_metric(failures, overall, "pass_rate", args.min_pass_rate)
    _check_metric(failures, overall, "route_ok", args.min_route_ok)
    _check_metric(failures, overall, "forbidden_clean@5", args.min_forbidden_clean)
    _check_metric(failures, overall, "context_reuse_ok", args.min_context_reuse_ok)

    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise SystemExit(f"Benchmark quality gate failed:\n{details}")

    print(
        "Benchmark quality gate passed: "
        f"turns={total_turns}, "
        f"pass_rate={_fmt(overall.get('pass_rate'))}, "
        f"route_ok={_fmt(overall.get('route_ok'))}, "
        f"forbidden_clean@5={_fmt(overall.get('forbidden_clean@5'))}, "
        f"context_reuse_ok={_fmt(overall.get('context_reuse_ok'))}"
    )


def _check_metric(failures: list[str], metrics: dict[str, Any], name: str, threshold: float) -> None:
    value = metrics.get(name)
    if value is None:
        failures.append(f"{name} is missing")
        return
    actual = float(value)
    if actual < threshold:
        failures.append(f"{name} {actual:.4f} < {threshold:.4f}")


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    main()
