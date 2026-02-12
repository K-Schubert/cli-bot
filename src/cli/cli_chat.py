# cli_chat5.py (updated)
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.append(_os.path.dirname(__file__))

try:
    __import__("pysqlite3")
    import sys as _sys2
    _sys2.modules["sqlite3"] = _sys2.modules.pop("pysqlite3")
except ImportError:
    pass  # fallback if pysqlite3 not installed

import argparse
import asyncio
import os
import sys
import glob
import shlex
import re
import json
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

from utils.utils import (
    EmbeddingClient,
    EmbeddingStore,
    LLMClient,
    TokenizerClient,
    ToolRegistry,
    ToolRunner,
    ConversationManager,
    RAGTool,
    SemanticSearchTool,
    MetadataSearchTool,
    read_developer_prompt,
    ZAS_SYSTEM_IDENTITY_FR,
    run_llm_round,
    RerankerClient,
    VLMParserClient,
    upsert_pdf_into_store,
    init_bm25,
    MemoryCompressor,
)


PDF_CMD_PATTERN = re.compile(r"(?:^|\s)/pdf\s+([^\s].*?)\s*$")


async def amain(args):
    # --- Resolve config from CLI/env ---
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

    # Vision (VLM) for PDF parsing
    vision_base = args.vision_base or os.environ.get("COPILOT_VISION_BASE_URL")
    if not vision_base:
        raise RuntimeError("Missing environment variable: COPILOT_VISION_BASE_URL")
    vision_model = args.vision_model or os.environ.get("COPILOT_VISION_MODEL", "copilot-inference-vision")

    # Reranker (env or CLI overrides)
    reranker_base = args.reranker_base or os.environ.get("RERANKER_BASE_URL")
    if not reranker_base:
        raise RuntimeError("Missing environment variable: RERANKER_BASE_URL")
    reranker_model = args.reranker_model or os.getenv("RERANKER_MODEL", "qwen-reranker")

    # --- Instantiate clients ---
    tokenizer = TokenizerClient(tokenizer_base, copilot_model)
    llm = LLMClient(copilot_base, api_key, copilot_model)
    embedder = EmbeddingClient(embedding_base, api_key, embedding_model)
    vlm = VLMParserClient(vision_base, api_key, vision_model)  # NEW

    # Optional reranker client
    reranker_client = None
    if args.doc_validation in ("rerank", "filter"):
        reranker_client = RerankerClient(
            base_url=reranker_base,
            model=reranker_model,
            timeout=args.reranker_timeout,
            instruction=args.reranker_instruction,
            threshold=args.doc_threshold,
        )

    # --- Vector store (Chroma) ---
    store = EmbeddingStore(collection_name=args.collection)
    match args.csv:
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

    # Init bm25 index once
    print("[index] Initializing BM25 index")
    init_bm25(store)

    # --- Tools ---
    registry = ToolRegistry()

    # semantic search tool (exposed to LLM)
    registry.register(
        SemanticSearchTool(
            store=store,
            reranker=reranker_client,
            validation_mode=args.doc_validation,
            verbose=args.verbose,
        ).build(embedder=embedder, n_results=args.search_k)
    )

    # metadata search tool (kept in code + registry, BUT NOT exposed to LLM yet)
    registry.register(
        MetadataSearchTool(
            store=store,
            reranker=reranker_client,
            validation_mode=args.doc_validation,
            verbose=args.verbose,
        ).build(limit=args.search_k)
    )

    tools = ToolRunner(registry)

    # --- System/Developer messages ---
    developer_prompt = read_developer_prompt(args.dev_prompt, response_format="Concise")

    # Expose ONLY semantic_search for now (hide metadata_search from the developer tool list)
    tool_descs_for_llm = registry.harmony_descriptions(exclude=["metadata_search"])

    conv = ConversationManager(
        language_system_identity=ZAS_SYSTEM_IDENTITY_FR,
        developer_prompt=developer_prompt,
        tools=registry,
        tool_descriptions=tool_descs_for_llm,  # NEW
    )

    # NEW: rolling memory compressor (extra tool-less LLM call after each completed turn)
    memory_compressor = MemoryCompressor(llm=llm)

    print("\nZAS/EAK-Copilot^GPT — CLI (type /exit to quit, /reset to clear, /pdf <path> to upload PDF)")

    # Optional: quick sanity check
    if args.sanity:
        print("[sanity] /tokenize Hello world! =>", tokenizer.tokenize("Hello world!"))
        print("[sanity] token ids =>", tokenizer.get_token_ids("Hello world!"))
        print("[sanity] /detokenize [1,2,3] =>", tokenizer.detokenize([1, 2, 3]))

    # --- Chat loop ---
    while True:
        try:
            raw = input("\n\033[1;92muser> \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("bye!")
            return

        if not raw:
            continue

        if raw == "/exit":
            print("\033[0mbye!")
            return
        if raw == "/reset":
            conv.reset_to_preamble(clear_memory=True)  # NEW: clears developer <memory>
            print("\033[0m(conversation reset)")
            continue

        # Handle inline /pdf command (anywhere in the line).
        # Accept quotes or spaces: e.g. /pdf /path/a.pdf  OR  /pdf "/path with spaces/a.pdf"
        pdf_match = None
        if "/pdf" in raw:
            # Robust extraction: take the last "/pdf ..." occurrence in case user typed multiple
            matches = list(re.finditer(r"/pdf\s+(.+?)(?=$|(?=\s/)|$)", raw))
            if matches:
                pdf_match = matches[-1]
        if pdf_match:
            pdf_arg = pdf_match.group(1).strip()
            # If quoted, shlex.split safely
            try:
                parts = shlex.split(pdf_arg)
                pdf_path = parts[0] if parts else pdf_arg
            except Exception:
                pdf_path = pdf_arg

            # Remove the "/pdf something" segment from the user utterance to keep the question
            start, end = pdf_match.span()
            question = (raw[:start] + raw[end:]).strip()
            if not question:
                question = "Fais un résumé de ce document et liste les points clés."

            # Perform upsert BEFORE adding the user question to the conversation
            try:
                trace = await upsert_pdf_into_store(
                    pdf_path=pdf_path,
                    embedder=embedder,
                    store=store,
                    vlm=vlm,
                    collection_name=args.collection,
                    max_pages=args.pdf_max_pages,
                )
                
                RAGTool(
                    store=store,
                    reranker=reranker_client,
                    validation_mode=args.doc_validation,
                    verbose=args.verbose,
                )._refresh_bm25()
                
                # Add a tool-trace so the LLM can scope search by metadata (pdf_name, etc.)
                conv.add_tool_output("pdf_upsert", json.dumps(trace, ensure_ascii=False))
                
                print(
                    f"\033[96m[upload] Upserted PDF with source: '{trace['source']}' \033[0m"
                    f"\033[96m({trace['page_count']} pages) as {trace['chunks']} chunks at {trace['upserted_at']}\033[0m"
                )
            except Exception as e:
                print(f"\033[96m[upload] Failed to upsert PDF: {e}\033[0m")
                # Continue anyway with the user's raw input
                question = raw

            # Now feed the cleaned question to the conversation
            conv.add_user(question)
        else:
            # Natural language only — pass the user's text as-is
            conv.add_user(raw)

        # Run a single model round with auto tools
        reply, n_tokens = await run_llm_round(
            conv=conv,
            llm=llm,
            tokenizer=tokenizer,
            tools=tools,
            temperature=args.temperature,
            max_tool_loops=3,
            debug_history=args.history,
        )

        print("\n\033[1;94massistant> \033[0m", reply)
        print(f"\n\33[35m[Context Window Usage] {n_tokens}/131'000 | {n_tokens/131_000*100:.2f}%")

        # --- NEW: Update rolling memory + strip traces (keep history clean) ---
        try:
            seg = conv.extract_last_turn_segment()
            new_memory_json = await memory_compressor.update_memory_json(
                prior_memory_json=conv.memory_block,
                turn_user=seg.get("user", ""),
                turn_assistant=seg.get("assistant", reply),
                turn_tools=seg.get("tools", []),
            )
            conv.set_memory_block(new_memory_json)

            # After memory is updated, remove ALL tool messages to avoid context bloat
            conv.strip_all_tool_traces()

        except Exception as e:
            # Fail-soft: never break the chat UX if memory compression fails
            print(f"\033[93m[memory] warning: failed to update memory ({e})\033[0m")
            # Still strip tool traces to keep context clean
            try:
                conv.strip_all_tool_traces()
            except Exception:
                pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CLI for agentic RAG (gpt-oss)")
    p.add_argument("--dev-prompt", required=True, help="Path to developer_prompt.md")
    p.add_argument("--csv", help="CSV with columns: content, embedding, metadata")
    p.add_argument("--collection", default="docs", help="Chroma collection name")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--sanity", action="store_true", help="Run tokenizer sanity checks on start")

    # NEW: search controls
    p.add_argument("--search-k", type=int, default=3, help="Top-k docs to retrieve from vector DB")

    # NEW: reranker / validation controls
    p.add_argument(
        "--doc-validation",
        choices=["off", "rerank", "filter"],
        default="off",
        help=(
            "Use external reranker to validate documents: "
            "off = disabled, rerank = reorder by score, filter = keep only 'yes'"
        ),
    )
    p.add_argument(
        "--doc-threshold",
        type=float,
        default=0.9,
        help="Threshold for 'yes' when filtering (if the service only returns scores)",
    )
    p.add_argument(
        "--reranker-base",
        help="Override RERANKER_BASE_URL (defaults to the value in your environment)",
    )
    p.add_argument(
        "--reranker-model",
        help="Override RERANKER_MODEL (default 'qwen-reranker')",
    )
    p.add_argument(
        "--reranker-timeout",
        type=float,
        default=30.0,
        help="HTTP timeout (seconds) for reranker calls",
    )
    p.add_argument(
        "--reranker-instruction",
        help=(
            "Optional instruction string injected into the reranker system/user prompt. "
            "Defaults to the working snippet's instruction."
        ),
    )

    # vision model controls
    p.add_argument(
        "--vision-base",
        help="Override COPILOT_VISION_BASE_URL (defaults to the value in your environment)",
    )
    p.add_argument(
        "--vision-model",
        help="Override COPILOT_VISION_MODEL (default 'copilot-vision')",
    )
    p.add_argument(
        "--pdf-max-pages",
        type=int,
        default=None,
        help="Limit number of pages sent to the VLM for parsing (None = all).",
    )
    # Debugging
    p.add_argument(
        "--verbose",
        type=bool,
        nargs="?",
        const=True,
        default=False,
        help="Print logs for debugging (default: False).",
    )

    # Debugging history
    p.add_argument(
        "--history",
        action="store_true",
        help="Print rendered prompt and Harmony messages sent to the LLM for each user query",
    )

    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("bye!")
        sys.exit(0)
