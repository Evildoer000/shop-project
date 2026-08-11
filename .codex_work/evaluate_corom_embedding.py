from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import BertModel, BertTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = (
    PROJECT_ROOT
    / "modelscope"
    / "models"
    / "iic--nlp_corom_sentence-embedding_chinese-base-ecom"
    / "snapshots"
    / "master"
)
DATASET_DIR = PROJECT_ROOT / "ecommerce_agent_dataset"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = BertModel.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        add_pooling_layer=False,
    ).to(device)
    model.eval()
    print(f"device={device}")

    official_query = "\u9614\u817f\u88e4\u5973\u51ac\u725b\u4ed4"
    official_encoded = tokenizer(
        [official_query],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    official_encoded = {
        key: value.to(device) for key, value in official_encoded.items()
    }
    with torch.inference_mode():
        official_hidden = model(**official_encoded).last_hidden_state
    attention_mask = official_encoded["attention_mask"].unsqueeze(-1)
    pooling_outputs = {
        "cls_raw": official_hidden[:, 0],
        "cls_normalized": torch.nn.functional.normalize(
            official_hidden[:, 0],
            p=2,
            dim=1,
        ),
        "mean_raw": (
            (official_hidden * attention_mask).sum(dim=1)
            / attention_mask.sum(dim=1).clamp(min=1)
        ),
    }
    pooling_outputs["mean_normalized"] = torch.nn.functional.normalize(
        pooling_outputs["mean_raw"],
        p=2,
        dim=1,
    )
    for pooling_name, pooling_output in pooling_outputs.items():
        print(
            f"official_{pooling_name}_first3="
            + ",".join(
                f"{value:.8f}"
                for value in pooling_output[0, :3].detach().cpu().tolist()
            )
        )
    print("official_expected_first3=-0.23219466,0.41309455,0.26903808")

    query = (
        "\u6211\u662f\u6cb9\u76ae\uff0c\u9884\u7b97150\u4ee5\u5185\uff0c"
        "\u63a8\u8350\u590f\u5929\u901a\u52e4\u4e0d\u95f7\u7684\u9632\u6652\u971c"
    )
    samples = [
        query,
        (
            "\u8587\u8bfa\u5a1c\u6e05\u900f\u9632\u6652\u4e73 \u654f\u611f\u808c "
            "\u9632\u6652\u971c \u6e05\u723d\u4fee\u62a4\u8f7b\u8584 "
            "\u590f\u5929\u901a\u52e4\u4e0d\u95f7 128\u5143"
        ),
        (
            "\u7fbd\u897f\u767d\u7389\u9632\u6652\u971c SPF50 "
            "\u8f7b\u8584\u901a\u52e4 \u6e05\u723d\u4e0d\u6cb9\u817b 189\u5143"
        ),
        "\u7537\u58eb\u8fd0\u52a8\u77ed\u889c \u5438\u6c57\u900f\u6c14 \u590f\u5b63\u8584\u6b3e",
        "\u9152\u5e97\u53cc\u4eba\u81ea\u52a9\u9910\u5957\u9910 \u5468\u672b\u901a\u7528",
    ]
    vectors = embed(model, tokenizer, samples, device=device)
    scores = vectors @ vectors[0]
    print("sample_scores=")
    for text, score in zip(samples, scores.tolist()):
        print(f"  {score:.6f}\t{text}")
    repeated = embed(model, tokenizer, [query], device=device)[0]
    print(f"same_text_cosine={float(vectors[0] @ repeated):.8f}")
    print(f"embedding_dim={vectors.shape[1]}")

    warmup = 5
    repeats = 50
    for _ in range(warmup):
        embed(model, tokenizer, [query], device=device)
    started = time.perf_counter()
    for _ in range(repeats):
        embed(model, tokenizer, [query], device=device)
    print(f"single_query_latency_ms={(time.perf_counter() - started) * 1000 / repeats:.2f}")

    products = load_products()
    lengths = [
        len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        for _, _, text in products
    ]
    print(f"product_count={len(products)}")
    print(
        "token_length="
        f"min:{min(lengths)},"
        f"median:{statistics.median(lengths):.0f},"
        f"p90:{percentile(lengths, 0.9):.0f},"
        f"p95:{percentile(lengths, 0.95):.0f},"
        f"max:{max(lengths)},"
        f"over128:{sum(length > 128 for length in lengths)}"
    )

    if args.full:
        run_full_retrieval(model, tokenizer, products, device=device)


def embed(
    model: BertModel,
    tokenizer: BertTokenizer,
    texts: list[str],
    *,
    device: str,
    max_length: int = 128,
) -> torch.Tensor:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model(**encoded).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(output, p=2, dim=1).cpu()


def load_products() -> list[tuple[str, str, str]]:
    result: list[tuple[str, str, str]] = []
    for category_dir in sorted(DATASET_DIR.glob("[1-4]_*")):
        if any(ord(char) > 127 for char in category_dir.name):
            continue
        for path in sorted((category_dir / "data").glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            product_id = str(data.get("product_id") or "")
            title = str(data.get("title") or "")
            text = compact_product_text(data)
            result.append((product_id, title, text))
    return result


def compact_product_text(data: dict) -> str:
    rag = data.get("rag_knowledge") or {}
    faq = rag.get("official_faq") or []
    reviews = rag.get("user_reviews") or []
    fields = [
        data.get("title"),
        data.get("brand"),
        data.get("category"),
        data.get("sub_category"),
        rag.get("marketing_description"),
        " ".join(
            f"{item.get('question', '')} {item.get('answer', '')}"
            for item in faq[:2]
        ),
        " ".join(str(item.get("content") or "") for item in reviews[:2]),
    ]
    return " ".join(str(value).strip() for value in fields if value)


def run_full_retrieval(
    model: BertModel,
    tokenizer: BertTokenizer,
    products: list[tuple[str, str, str]],
    *,
    device: str,
) -> None:
    started = time.perf_counter()
    vectors: list[torch.Tensor] = []
    batch_size = 64 if device == "cuda" else 8
    for start in range(0, len(products), batch_size):
        batch = [item[2] for item in products[start : start + batch_size]]
        vectors.append(embed(model, tokenizer, batch, device=device))
        if (start // batch_size + 1) % 25 == 0:
            print(f"embedded={min(start + batch_size, len(products))}/{len(products)}")
    matrix = torch.cat(vectors, dim=0)
    print(f"full_embedding_seconds={time.perf_counter() - started:.2f}")

    queries = [
        (
            "\u9632\u6652",
            "\u6211\u662f\u6cb9\u76ae\uff0c\u9884\u7b97150\u4ee5\u5185\uff0c"
            "\u63a8\u8350\u590f\u5929\u901a\u52e4\u4e0d\u95f7\u7684\u9632\u6652\u971c",
        ),
        (
            "\u8033\u673a",
            "\u9884\u7b97300\u5143\uff0c\u60f3\u8981\u964d\u566a\u597d\u7684"
            "\u65e0\u7ebf\u84dd\u7259\u8033\u673a\uff0c\u4e0a\u4e0b\u73ed\u5730\u94c1\u7528",
        ),
        (
            "\u8dd1\u978b",
            "\u63a8\u8350\u9002\u5408\u590f\u5929\u8dd1\u6b65\u7684\u900f\u6c14"
            "\u8f7b\u4fbf\u8fd0\u52a8\u978b\uff0c\u9884\u7b97500\u4ee5\u5185",
        ),
        (
            "\u9762\u971c",
            "\u654f\u611f\u5e72\u76ae\u79cb\u51ac\u4fdd\u6e7f\u4fee\u62a4\u9762\u971c\uff0c"
            "\u4e0d\u8981\u592a\u9999\uff0c\u9884\u7b97200\u5143",
        ),
        (
            "\u96f6\u98df",
            "\u51cf\u8102\u671f\u529e\u516c\u5ba4\u89e3\u998b\u7684"
            "\u4f4e\u7cd6\u9ad8\u86cb\u767d\u96f6\u98df",
        ),
    ]
    query_vectors = embed(
        model,
        tokenizer,
        [query for _, query in queries],
        device=device,
    )
    for (name, query), query_vector in zip(queries, query_vectors):
        scores = matrix @ query_vector
        top_scores, top_indices = torch.topk(scores, k=10)
        print(f"query={name}\t{query}")
        for rank, (score, index) in enumerate(
            zip(top_scores.tolist(), top_indices.tolist()),
            start=1,
        ):
            product_id, title, _ = products[index]
            print(f"  {rank:02d}\t{score:.6f}\t{product_id}\t{title}")


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return float(ordered[index])


if __name__ == "__main__":
    main()
