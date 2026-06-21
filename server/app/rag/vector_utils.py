from __future__ import annotations


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """两个归一化向量的余弦相似度 = 点积。"""
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))
