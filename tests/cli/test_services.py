from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cli import services as cli_services
from utils.utils import Tool


class DummySemanticSearchTool:
    def __init__(self, *args, **kwargs):
        pass

    def build(self, *, embedder, n_results):
        return Tool(
            name="semantic_search",
            description="",
            parameters={},
            func=lambda **_: None,
        )


class DummyMetadataSearchTool:
    def __init__(self, *args, **kwargs):
        pass

    def build(self, *, limit):
        return Tool(
            name="metadata_search",
            description="",
            parameters={},
            func=lambda **_: None,
        )


def test_build_tooling_registers_tools(monkeypatch):
    monkeypatch.setattr(cli_services, "SemanticSearchTool", DummySemanticSearchTool)
    monkeypatch.setattr(cli_services, "MetadataSearchTool", DummyMetadataSearchTool)

    store = SimpleNamespace()
    clients = SimpleNamespace(embedder=object(), reranker=None)

    bundle = cli_services.build_tooling(
        store=store,
        clients=clients,
        doc_validation="off",
        search_k=3,
        verbose=False,
    )

    assert bundle.registry.get("semantic_search").name == "semantic_search"
    assert bundle.registry.get("metadata_search").name == "metadata_search"
    assert all(desc for desc in bundle.tool_descriptions)