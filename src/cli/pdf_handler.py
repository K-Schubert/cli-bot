from __future__ import annotations

import json
import re
import shlex
from typing import Optional, Tuple

from utils.utils import RAGTool, upsert_pdf_into_store

from .constants import PDF_CMD_PATTERN

DEFAULT_PDF_PROMPT = "OCR et extrait les points clés du document."


def _find_pdf_match(raw: str) -> Optional[re.Match[str]]:
    if "/pdf" not in raw:
        return None
    matches = list(PDF_CMD_PATTERN.finditer(raw))
    return matches[-1] if matches else None


def extract_pdf_command(raw: str) -> Tuple[Optional[str], str]:
    match = _find_pdf_match(raw)
    if not match:
        return None, raw

    pdf_arg = match.group(1).strip()
    try:
        parts = shlex.split(pdf_arg)
        pdf_path = parts[0] if parts else pdf_arg
    except Exception:
        pdf_path = pdf_arg

    start, end = match.span()
    question = (raw[:start] + raw[end:]).strip()
    if not question:
        question = DEFAULT_PDF_PROMPT

    return pdf_path, question


async def process_pdf_upload(
    *,
    pdf_path: str,
    embedder,
    store,
    vlm,
    collection_name: str,
    reranker,
    doc_validation: str,
    verbose: bool,
    max_pages: Optional[int] = None,
) -> dict:
    trace = await upsert_pdf_into_store(
        pdf_path=pdf_path,
        embedder=embedder,
        store=store,
        vlm=vlm,
        collection_name=collection_name,
        max_pages=max_pages,
    )

    RAGTool(
        store=store,
        reranker=reranker,
        validation_mode=doc_validation,
        verbose=verbose,
    )._refresh_bm25()

    return trace


def format_pdf_trace(trace: dict) -> str:
    return json.dumps(trace, ensure_ascii=False)
