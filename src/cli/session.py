from __future__ import annotations

import asyncio

import traceback

from utils.utils import run_llm_round

from .constants import ASSISTANT_PROMPT, INTRO_BANNER, USER_PROMPT
from .conversation import build_conversation_manager, build_memory_compressor
from .pdf_handler import extract_pdf_command, format_pdf_trace, process_pdf_upload


def _print_sanity_checks(tokenizer) -> None:
    print("[sanity] /tokenize Hello world! =>", tokenizer.tokenize("Hello world!"))
    print("[sanity] token ids =>", tokenizer.get_token_ids("Hello world!"))
    print("[sanity] /detokenize [1,2,3] =>", tokenizer.detokenize([1, 2, 3]))


async def _update_memory(conv, memory_compressor, reply: str) -> None:
    segment = conv.extract_last_turn_segment()
    new_memory_json = await memory_compressor.update_memory_json(
        prior_memory_json=conv.memory_block,
        turn_user=segment.get("user", ""),
        turn_assistant=segment.get("assistant", reply),
        turn_tools=segment.get("tools", []),
    )
    conv.set_memory_block(new_memory_json)
    conv.strip_all_tool_traces()


async def run_session(
    *,
    args,
    clients,
    store,
    tooling,
) -> None:
    conversation = build_conversation_manager(
        developer_prompt_path=args.dev_prompt,
        tool_registry=tooling.registry,
        tool_descriptions=tooling.tool_descriptions,
    )
    memory_compressor = build_memory_compressor(clients.llm)

    print(INTRO_BANNER)

    if args.sanity:
        _print_sanity_checks(clients.tokenizer)

    while True:
        try:
            raw = input(USER_PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print("bye!")
            return

        if not raw:
            continue

        if raw == "/exit":
            print("\033[0mbye!")
            return
        if raw == "/reset":
            conversation.reset_to_preamble(clear_memory=True)
            print("\033[0m(conversation reset)")
            continue

        pdf_path, question = extract_pdf_command(raw)
        if pdf_path:
            try:
                trace = await process_pdf_upload(
                    pdf_path=pdf_path,
                    embedder=clients.embedder,
                    store=store,
                    vlm=clients.vlm,
                    collection_name=args.collection,
                    reranker=clients.reranker,
                    doc_validation=args.doc_validation,
                    verbose=args.verbose,
                    max_pages=args.pdf_max_pages,
                )
                conversation.add_tool_output("pdf_upsert", format_pdf_trace(trace))
                print(
                    f"\033[96m[upload] Upserted PDF with source: '{trace['source']}' \033[0m"
                    f"\033[96m({trace['page_count']} pages) as {trace['chunks']} chunks at {trace['upserted_at']}\033[0m"
                )
                conversation.add_user(question)
            except Exception as exc:
                print(f"\033[96m[upload] Failed to upsert PDF: {exc}\033[0m")
                conversation.add_user(raw)
        else:
            conversation.add_user(raw)

        reply, n_tokens = await run_llm_round(
            conv=conversation,
            llm=clients.llm,
            tokenizer=clients.tokenizer,
            tools=tooling.runner,
            temperature=args.temperature,
            max_tool_loops=3,
            debug_history=args.history,
        )

        print(ASSISTANT_PROMPT, reply)
        print(f"\n\33[35m[Context Window Usage] {n_tokens}/131'000 | {n_tokens/131_000*100:.2f}%")

        try:
            await _update_memory(conversation, memory_compressor, reply)
        except Exception as exc:
            print(f"\033[93m[memory] warning: failed to update memory ({exc})\033[0m")
            traceback.print_exc()
            try:
                conversation.strip_all_tool_traces()
            except Exception:
                pass


async def run_async_entrypoint(args, clients, store, tooling) -> None:
    await run_session(
        args=args,
        clients=clients,
        store=store,
        tooling=tooling,
    )


def run_entrypoint(args, clients, store, tooling) -> None:
    asyncio.run(
        run_async_entrypoint(
            args,
            clients,
            store,
            tooling,
        )
    )
