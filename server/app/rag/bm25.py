from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.db.models import Product

import threading
import jieba

_JIEBA_INIT_LOCK = threading.Lock()
_JIEBA_READY = False

# 电商场景词典: 让 jieba 把这些词当整体, 不切碎
_DOMAIN_DICT = [
    "跑鞋", "跑步鞋", "长跑", "慢跑",
    "篮球鞋", "徒步鞋", "登山鞋",
    "速干T恤", "速干", "瑜伽裤", "户外裤",
    "精华液", "化妆水", "面霜", "防晒霜",
    "洗面奶", "眼霜", "卸妆油", "唇釉",
    "智能手机", "笔记本电脑", "平板电脑",
    "真无线耳机", "折叠屏",
    "方便食品", "功能饮料", "碳酸饮料",
    "特步", "安踏", "李宁", "耐克", "阿迪达斯",
    "华为", "小米", "苹果",
    "三顿半", "元气森林", "农夫山泉",
    "雅诗兰黛", "兰蔻", "科颜氏", "薇诺娜", "理肤泉",
]


def _ensure_jieba_ready():
    global _JIEBA_READY
    if _JIEBA_READY:
        return
    with _JIEBA_INIT_LOCK:
        if _JIEBA_READY:
            return
        jieba.setLogLevel(60)
        jieba.initialize()
        for word in _DOMAIN_DICT:
            jieba.add_word(word, freq=10000)
        _JIEBA_READY = True



@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75


class ProductBM25Scorer:
    def __init__(self, products: list[Product], config: BM25Config | None = None) -> None:
        self.products = products
        self.config = config or BM25Config()
        self.product_ids = [product.product_id for product in products]
        self.documents = [self._document_tokens(product) for product in products]
        self.doc_lengths = [len(document) for document in self.documents]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.doc_freqs = self._doc_freqs(self.documents)

    def score(self, query: str, top_k: int | None = None) -> dict[str, float]:
        if not self.products:
            return {}
        query_terms = self._query_tokens(query)
        if not query_terms:
            return {}

        raw_scores: dict[str, float] = {}
        for product_id, document, doc_length in zip(self.product_ids, self.documents, self.doc_lengths, strict=True):
            score = self._score_document(query_terms, document, doc_length)
            if score > 0:
                raw_scores[product_id] = score

        if not raw_scores:
            return {}

        max_score = max(raw_scores.values())
        normalized = {product_id: score / max_score for product_id, score in raw_scores.items()}
        ranked = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return dict(ranked)

    def _score_document(self, query_terms: Counter[str], document: Counter[str], doc_length: int) -> float:
        score = 0.0
        total_docs = len(self.documents)
        for term, query_count in query_terms.items():
            term_freq = document.get(term, 0)
            if term_freq == 0:
                continue
            doc_freq = self.doc_freqs.get(term, 0)
            idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            denominator = term_freq + self.config.k1 * (
                1 - self.config.b + self.config.b * doc_length / max(1.0, self.avg_doc_length)
            )
            score += query_count * idf * (term_freq * (self.config.k1 + 1)) / denominator
        return score

    def _document_tokens(self, product: Product) -> Counter[str]:
        tokens: list[str] = []
        weighted_fields = [
            (product.name, 3),
            (product.category, 3),
            (product.sub_category or "", 4),
            (product.brand, 3),
            (" ".join(str(tag) for tag in product.tags or []), 3),
            (product.description, 1),
            (product.review_summary, 1),
            (product.search_text(), 1),
        ]
        for text, weight in weighted_fields:
            field_tokens = tokenize_for_bm25(text)
            for _ in range(weight):
                tokens.extend(field_tokens)
        return Counter(tokens)

    def _query_tokens(self, query: str) -> Counter[str]:
        return Counter(tokenize_for_bm25(query))

    def _doc_freqs(self, documents: list[Counter[str]]) -> dict[str, int]:
        freqs: dict[str, int] = {}
        for document in documents:
            for term in document:
                freqs[term] = freqs.get(term, 0) + 1
        return freqs


def tokenize_for_bm25(text: str) -> list[str]:
    normalized = text.lower()
    tokens: list[str] = []
    tokens.extend(_latin_tokens(normalized))
    tokens.extend(_numeric_tokens(normalized))
    tokens.extend(_chinese_ngrams(normalized))
    return [token for token in tokens if token not in _stop_terms()]


def _latin_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9+%-]{1,}", text)


def _numeric_tokens(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?(?:ml|g|l|gb|tb|英寸|寸|元|%|颗|盒|罐|瓶|袋)?", text)


def _chinese_ngrams(text: str) -> list[str]:
    """jieba 中文分词 (替代旧的字符 N-gram).
    
    保留函数名以兼容外部 import; 内部用 jieba.cut 切真词.
    """
    _ensure_jieba_ready()
    tokens = []
    for chunk in re.findall(r"[一-鿿]+", text):
        for word in jieba.cut(chunk):
            word = word.strip()
            if len(word) > 1:    # 过滤单字 (jieba 切出的"的""了""是" 等)
                tokens.append(word)
    return tokens


def _stop_terms() -> set[str]:
    return {
        "一个",
        "一点",
        "一些",
        "左右",
        "以内",
        "以下",
        "以上",
        "可以",
        "想买",
        "想找",
        "最好",
        "预算",
        "适合",
        "都行",
        "都可以",
        "帮我",
        "推荐",
    }
