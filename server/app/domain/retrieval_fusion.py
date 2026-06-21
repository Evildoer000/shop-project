from __future__ import annotations


def rrf_fuse(
    ranked_lists: list[list[str]],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[str]:
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0

    for ranked_ids in ranked_lists:
        seen_in_list: set[str] = set()
        for rank, item_id in enumerate(ranked_ids, start=1):
            if not item_id or item_id in seen_in_list:
                continue
            seen_in_list.add(item_id)
            if item_id not in first_seen:
                first_seen[item_id] = order
                order += 1
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank)

    ranked = sorted(
        scores,
        key=lambda item_id: (-scores[item_id], first_seen[item_id]),
    )
    return ranked[:top_k]
