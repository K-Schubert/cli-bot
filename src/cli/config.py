from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from importlib import import_module
from typing import Optional


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI for agentic RAG (gpt-oss)")
    parser.add_argument("--dev-prompt", required=True, help="Path to developer_prompt.md")
    parser.add_argument("--csv", help="CSV with columns: content, embedding, metadata")
    parser.add_argument("--collection", default="docs", help="Chroma collection name")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sanity", action="store_true", help="Run tokenizer sanity checks on start")
    parser.add_argument("--search-k", type=int, default=3, help="Top-k docs to retrieve from vector DB")
    parser.add_argument(
        "--doc-validation",
        choices=["off", "rerank", "filter"],
        default="off",
        help=(
            "Use external reranker to validate documents: "
            "off = disabled, rerank = reorder by score, filter = keep only 'yes'"
        ),
    )
    parser.add_argument(
        "--doc-threshold",
        type=float,
        default=0.9,
        help="Threshold for 'yes' when filtering (if the service only returns scores)",
    )
    parser.add_argument(
        "--reranker-base",
        help="Override RERANKER_BASE_URL (defaults to the value in your environment)",
    )
    parser.add_argument(
        "--reranker-model",
        help="Override RERANKER_MODEL (default 'qwen-reranker')",
    )
    parser.add_argument(
        "--reranker-timeout",
        type=float,
        default=30.0,
        help="HTTP timeout (seconds) for reranker calls",
    )
    parser.add_argument(
        "--reranker-instruction",
        help=(
            "Optional instruction string injected into the reranker system/user prompt. "
            "Defaults to the working snippet's instruction."
        ),
    )
    parser.add_argument(
        "--vision-base",
        help="Override COPILOT_VISION_BASE_URL (defaults to the value in your environment)",
    )
    parser.add_argument(
        "--vision-model",
        help="Override COPILOT_VISION_MODEL (default 'copilot-vision')",
    )
    parser.add_argument(
        "--pdf-max-pages",
        type=int,
        default=None,
        help="Limit number of pages sent to the VLM for parsing (None = all).",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        nargs="?",
        const=True,
        default=False,
        help="Print logs for debugging (default: False).",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Print rendered prompt and Harmony messages sent to the LLM for each user query",
    )

    return parser.parse_args(argv)


@dataclass
class ServiceSettings:
    api_key: str
    copilot_base: str
    copilot_model: str
    embedding_base: str
    embedding_model: str
    tokenizer_base: str
    vision_base: str
    vision_model: str
    reranker_base: Optional[str]
    reranker_model: Optional[str]


def load_service_settings(args: argparse.Namespace) -> ServiceSettings:
    try:
        load_dotenv = getattr(import_module("dotenv"), "load_dotenv")
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard only
        raise RuntimeError("python-dotenv is required to load environment variables") from exc

    load_dotenv()

    api_key = os.environ.get("API_KEY", "NULL")

    copilot_base = os.environ.get("COPILOT_BASE_URL")
    if not copilot_base:
        raise RuntimeError("Missing environment variable: COPILOT_BASE_URL")
    copilot_model = os.environ.get("COPILOT_MODEL", "copilot-inference")

    embedding_base = os.environ.get("EMBEDDING_BASE_URL")
    if not embedding_base:
        raise RuntimeError("Missing environment variable: EMBEDDING_BASE_URL")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "qwen-embedding")

    tokenizer_base = os.environ.get("TOKENIZER_BASE_URL")
    if not tokenizer_base:
        raise RuntimeError("Missing environment variable: TOKENIZER_BASE_URL")

    vision_base = args.vision_base or os.environ.get("COPILOT_VISION_BASE_URL")
    if not vision_base:
        raise RuntimeError("Missing environment variable: COPILOT_VISION_BASE_URL")
    vision_model = args.vision_model or os.environ.get("COPILOT_VISION_MODEL", "copilot-inference-vision")

    reranker_base = args.reranker_base or os.environ.get("RERANKER_BASE_URL")
    if not reranker_base and args.doc_validation in ("rerank", "filter"):
        raise RuntimeError("Missing environment variable: RERANKER_BASE_URL")
    reranker_model = args.reranker_model or os.getenv("RERANKER_MODEL", "qwen-reranker")

    return ServiceSettings(
        api_key=api_key,
        copilot_base=copilot_base,
        copilot_model=copilot_model,
        embedding_base=embedding_base,
        embedding_model=embedding_model,
        tokenizer_base=tokenizer_base,
        vision_base=vision_base,
        vision_model=vision_model,
        reranker_base=reranker_base,
        reranker_model=reranker_model,
    )
