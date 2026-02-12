from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Optional

from utils.utils import (
    EmbeddingClient,
    EmbeddingStore,
    LLMClient,
    MetadataSearchTool,
    RerankerClient,
    SemanticSearchTool,
    TokenizerClient,
    ToolRegistry,
    ToolRunner,
    VLMParserClient,
    init_bm25,
)

from .config import ServiceSettings


@dataclass
class ClientBundle:
    tokenizer: TokenizerClient
    llm: LLMClient
    embedder: EmbeddingClient
    vlm: VLMParserClient
    reranker: Optional[RerankerClient]


@dataclass
class ToolingBundle:
    registry: ToolRegistry
    runner: ToolRunner
    tool_descriptions: list


def build_clients(settings: ServiceSettings, args) -> ClientBundle:
    tokenizer = TokenizerClient(settings.tokenizer_base, settings.copilot_model)
    llm = LLMClient(settings.copilot_base, settings.api_key, settings.copilot_model)
    embedder = EmbeddingClient(settings.embedding_base, settings.api_key, settings.embedding_model)
    vlm = VLMParserClient(settings.vision_base, settings.api_key, settings.vision_model)

    reranker_client: Optional[RerankerClient] = None
    if args.doc_validation in ("rerank", "filter") and settings.reranker_base:
        reranker_client = RerankerClient(
            base_url=settings.reranker_base,
            model=settings.reranker_model or "qwen-reranker",
            timeout=args.reranker_timeout,
            instruction=args.reranker_instruction,
            threshold=args.doc_threshold,
        )

    return ClientBundle(
        tokenizer=tokenizer,
        llm=llm,
        embedder=embedder,
        vlm=vlm,
        reranker=reranker_client,
    )


def prepare_store(collection: str, csv_arg: Optional[str], *, verbose: bool = False) -> EmbeddingStore:
    store = EmbeddingStore(collection_name=collection)

    match csv_arg:
        case "all":
            for path in glob.glob("/opt/app-root/src/data/*.csv"):
                print(f"[index] Upserting CSV: {path}")
                store.upsert_from_csv(path)
        case csv_path if csv_path and os.path.isfile(csv_path) and csv_path.endswith(".csv"):
            print(f"[index] Upserting CSV: {csv_path}")
            store.upsert_from_csv(csv_path)
        case _:
            # allow empty CSV param when relying on /pdf uploads only
            pass

    print("[index] Initializing BM25 index")
    init_bm25(store)

    return store


def build_tooling(
    *,
    store: EmbeddingStore,
    clients: ClientBundle,
    doc_validation: str,
    search_k: int,
    verbose: bool,
) -> ToolingBundle:
    registry = ToolRegistry()

    semantic_tool = SemanticSearchTool(
        store=store,
        reranker=clients.reranker,
        validation_mode=doc_validation,
        verbose=verbose,
    ).build(embedder=clients.embedder, n_results=search_k)

    metadata_tool = MetadataSearchTool(
        store=store,
        reranker=clients.reranker,
        validation_mode=doc_validation,
        verbose=verbose,
    ).build(limit=search_k)

    registry.register(semantic_tool)
    registry.register(metadata_tool)

    runner = ToolRunner(registry)
    tool_descs = registry.harmony_descriptions(exclude=["metadata_search"])

    return ToolingBundle(registry=registry, runner=runner, tool_descriptions=tool_descs)
