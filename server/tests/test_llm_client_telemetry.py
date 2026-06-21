import json
import time

from app.services.llm_client import LlmClient


def test_llm_client_records_usage_latency_and_cost() -> None:
    client = LlmClient(component="UnitTest")
    client.settings.llm_model = "demo-model"
    client.settings.llm_base_url = "https://example.test/v1"
    client.settings.llm_input_price_per_1k = 0.01
    client.settings.llm_output_price_per_1k = 0.02

    started = time.perf_counter()
    client._record_telemetry(
        operation="unit.operation",
        stream=False,
        status="succeeded",
        started=started,
        attempt_count=1,
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    )

    payload = client.telemetry_payloads()[0]
    assert payload["component"] == "UnitTest"
    assert payload["model"] == "demo-model"
    assert payload["operation"] == "unit.operation"
    assert payload["usage"]["total_tokens"] == 150
    assert payload["estimated_cost"] == 0.002
    assert payload["latency_ms"] >= 0


def test_llm_client_stream_parser_keeps_usage_chunk() -> None:
    client = LlmClient(component="UnitTest")
    line = "data: " + json.dumps(
        {
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
    )

    data = client._stream_data_from_line(line)

    assert isinstance(data, dict)
    assert client._stream_delta_from_data(data) is None
    assert client._usage_from_response(data)["total_tokens"] == 5
