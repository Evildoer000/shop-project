import asyncio

from app.core.config import Settings
from app.services.image_attribute_extractor import ImageAttributeExtractor


def test_image_attribute_extractor_uses_ark_multimodal_payload(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake image bytes")
    calls = []

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"available":true,"category_guess":"服饰鞋包",'
                                '"product_type_guess":"外套","colors":["米白色"],'
                                '"style_tags":["通勤","简约"],"material_guess":"针织或羊毛混纺",'
                                '"occasion_tags":["春秋"],"retrieval_query":"米白色 简约通勤 外套",'
                                '"confidence":0.78,"uncertainty_note":"材质仅根据图片推测"}'
                            )
                        }
                    }
                ]
            }

    class FakeAsyncClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def post(self, url: str, headers: dict, json: dict) -> FakeResponse:
            calls.append({"url": url, "headers": headers, "json": json, "timeout": self.timeout})
            return FakeResponse()

    monkeypatch.setattr(
        "app.services.image_attribute_extractor.get_settings",
        lambda: Settings(
            vlm_api_key="sk-test",
            vlm_base_url="https://ark.cn-beijing.volces.com/api/v3",
            vlm_model="ep-20260514111645-lmgt2",
            vlm_timeout_seconds=12,
        ),
    )
    monkeypatch.setattr("app.services.image_attribute_extractor.httpx.AsyncClient", FakeAsyncClient)

    attributes = asyncio.run(ImageAttributeExtractor().extract(image_path, "帮我找相似款"))

    assert attributes.available is True
    assert attributes.product_type_guess == "外套"
    assert attributes.colors == ["米白色"]
    assert attributes.retrieval_query == "米白色 简约通勤 外套"
    assert calls[0]["url"] == "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-test"
    assert calls[0]["json"]["model"] == "ep-20260514111645-lmgt2"
    image_part = calls[0]["json"]["messages"][1]["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_image_attribute_extractor_unconfigured_returns_unavailable(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake image bytes")
    monkeypatch.setattr(
        "app.services.image_attribute_extractor.get_settings",
        lambda: Settings(vlm_api_key=None),
    )

    attributes = asyncio.run(ImageAttributeExtractor().extract(image_path, ""))

    assert attributes.available is False
    assert "not configured" in attributes.uncertainty_note


def test_image_attribute_extractor_hard_timeout_returns_unavailable(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake image bytes")
    monkeypatch.setattr(
        "app.services.image_attribute_extractor.get_settings",
        lambda: Settings(
            vlm_api_key="sk-test",
            vlm_base_url="https://ark.cn-beijing.volces.com/api/v3",
            vlm_model="ep-20260514111645-lmgt2",
            vlm_timeout_seconds=0.01,
            vlm_max_retries=0,
        ),
    )

    async def slow_extract_once(self, image_path, user_text):
        await asyncio.sleep(1)

    monkeypatch.setattr(ImageAttributeExtractor, "_extract_once", slow_extract_once)

    extractor = ImageAttributeExtractor()
    attributes = asyncio.run(extractor.extract(image_path, ""))

    assert attributes.available is False
    assert "VLM extraction failed" in attributes.uncertainty_note
    assert extractor.last_call is not None
    assert extractor.last_call.error_type == "TimeoutError"
