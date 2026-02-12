from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cli import config as cli_config


def _install_fake_dotenv(monkeypatch) -> None:
    fake_module = ModuleType("dotenv")
    fake_module.load_dotenv = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dotenv", fake_module)


def test_load_service_settings_reads_environment(monkeypatch):
    _install_fake_dotenv(monkeypatch)

    monkeypatch.setenv("COPILOT_BASE_URL", "https://copilot")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embedding")
    monkeypatch.setenv("TOKENIZER_BASE_URL", "https://tokenizer")
    monkeypatch.setenv("COPILOT_VISION_BASE_URL", "https://vision")

    args = SimpleNamespace(
        vision_base=None,
        vision_model=None,
        reranker_base=None,
        reranker_model=None,
        reranker_timeout=30.0,
        reranker_instruction=None,
        doc_threshold=0.9,
        doc_validation="off",
    )

    settings = cli_config.load_service_settings(args)

    assert settings.copilot_base == "https://copilot"
    assert settings.embedding_base == "https://embedding"
    assert settings.tokenizer_base == "https://tokenizer"
    assert settings.vision_base == "https://vision"


def test_load_service_settings_requires_reranker_when_enabled(monkeypatch):
    _install_fake_dotenv(monkeypatch)

    monkeypatch.setenv("COPILOT_BASE_URL", "https://copilot")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embedding")
    monkeypatch.setenv("TOKENIZER_BASE_URL", "https://tokenizer")
    monkeypatch.setenv("COPILOT_VISION_BASE_URL", "https://vision")

    args = SimpleNamespace(
        vision_base=None,
        vision_model=None,
        reranker_base=None,
        reranker_model=None,
        reranker_timeout=30.0,
        reranker_instruction=None,
        doc_threshold=0.9,
        doc_validation="filter",
    )

    with pytest.raises(RuntimeError):
        cli_config.load_service_settings(args)
