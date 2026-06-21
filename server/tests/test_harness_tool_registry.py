import pytest

from app.harness.tool_registry import ToolRegistry


class DummyTool:
    pass


def test_tool_registry_register_get_describe_and_require() -> None:
    registry = ToolRegistry()
    tool = DummyTool()

    registry.register("product_search", tool, description="文字商品检索原子能力")

    assert registry.get("product_search") is tool
    assert registry.require("product_search", DummyTool) is tool
    assert registry.describe() == [
        {
            "name": "product_search",
            "kind": "tool",
            "owner": "Orchestrator",
            "description": "文字商品检索原子能力",
        }
    ]


def test_tool_registry_rejects_duplicate_and_missing_tools() -> None:
    registry = ToolRegistry()
    registry.register("profile_lookup", DummyTool())

    with pytest.raises(ValueError, match="already registered"):
        registry.register("profile_lookup", DummyTool())
    with pytest.raises(KeyError, match="not registered"):
        registry.get("missing")
    with pytest.raises(TypeError, match="expected str"):
        registry.require("profile_lookup", str)
