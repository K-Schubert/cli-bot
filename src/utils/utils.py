# utils5.py
from __future__ import annotations

try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # fallback if pysqlite3 not installed

import re
import asyncio
import ast
import base64
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union
import uuid

import chromadb
import numpy as np
import pandas as pd
import requests
import httpx
from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi

# Optional deps for PDF->image
try:
    import fitz  # PyMuPDF
except Exception as _e_fitz:  # pragma: no cover
    fitz = None

try:
    from PIL import Image
except Exception as _e_pil:  # pragma: no cover
    Image = None

# Harmony SDK (as used in the notebook)
try:
    from openai_harmony import (
        Author,
        Conversation,
        DeveloperContent,
        HarmonyEncodingName,
        Message,
        Role,
        SystemContent,
        ToolDescription,
        load_harmony_encoding,
        ReasoningEffort,
    )
except Exception as e:  # pragma: no cover
    raise ImportError(
        "This module requires `openai-harmony` (same package you used in the notebook)."
    ) from e


# ------------------------
# Networking clients
# ------------------------


def _ca_bundle_path() -> str:
    return os.getenv("CA_BUNDLE_PATH", "/etc/ssl/certs/ca-certificates.crt")


def _llm_timeout(default: float = 600.0) -> float:
    raw = os.getenv("LLM_HTTP_TIMEOUT")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _build_requests_session() -> requests.Session:
    session = requests.Session()
    session.verify = _ca_bundle_path()
    session.trust_env = False
    return session


class TokenizerClient:
    """Minimal wrapper for vLLM-compatible /tokenize and /detokenize endpoints."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout: float = 30.0,
        http_client: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        if http_client is None:
            http_client = _build_requests_session()
        self.http_client = http_client

    def tokenize(self, prompt: str) -> Dict[str, Any]:
        url = f"{self.base_url}/tokenize"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = self.http_client.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def detokenize(self, token_ids: List[int]) -> Dict[str, Any]:
        url = f"{self.base_url}/detokenize"
        headers = {"Content-Type": "application/json"}
        payload = {"model": self.model_name, "tokens": token_ids}
        resp = self.http_client.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def count_tokens(self, prompt: str) -> Optional[int]:
        data = self.tokenize(prompt)
        return data.get("usage", {}).get("total_tokens")

    def get_token_ids(self, prompt: str) -> Optional[List[int]]:
        data = self.tokenize(prompt)
        return data.get("tokens")


class EmbeddingClient:
    """OpenAI-compatible embeddings client."""

    def __init__(self, base_url: str, api_key: str, model: str):
        http_client = httpx.AsyncClient(
            timeout=_llm_timeout(),
            trust_env=False,
            verify=_ca_bundle_path(),
        )
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
        )
        self.model = model

    async def embed(self, text: str) -> List[float]:
        resp = await self.client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        resp = await self.client.embeddings.create(model=self.model, input=texts)
        # Ensure same ordering
        return [d.embedding for d in resp.data]


class LLMClient:
    """OpenAI-compatible responses client (supports tool calling)."""

    def __init__(self, base_url: str, api_key: str, model: str):
        http_client = httpx.AsyncClient(
            timeout=_llm_timeout(),
            trust_env=False,
            verify=_ca_bundle_path(),
        )
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
        )
        self.model = model

    async def respond(self, prompt: str, *, temperature: float = 0.0, tool_choice: str = "auto"):
        # Mirrors your notebook usage of `responses.create`
        return await self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=temperature,
            tool_choice=tool_choice,
            stream=False,
        )


# ------------------------
# NEW: Vision parser client (PDF -> Markdown via VLM)
# ------------------------

class VLMParserClient:
    """
    Minimal chat-completions compatible VLM client for parsing images to text.

    We post to {base_url}/chat/completions with:
      {
        "model": model,
        "messages": [{role, content: [ {type:"text"},{type:"image_url", image_url:{url:...}}, ... ]}],
        "temperature": 0
      }

    If that fails, we best-effort try {base_url}/responses with an equivalent payload.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        http_client: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        if http_client is None:
            http_client = _build_requests_session()
        self.http_client = http_client

    @staticmethod
    def _pdf_to_data_urls(path: str, *, dpi: int = 300, max_pages: Optional[int] = None) -> Tuple[List[str], int]:
        if fitz is None or Image is None:
            raise RuntimeError(
                "PDF parsing requires PyMuPDF (`fitz`) and Pillow (`PIL`). Please install them."
            )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No such PDF: {path}")

        doc = fitz.open(path)
        n_pages = doc.page_count
        count = n_pages if max_pages is None else min(n_pages, max_pages)
        urls: List[str] = []

        for i in range(count):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=dpi)
            image = Image.open(BytesIO(pix.tobytes("png")))
            buf = BytesIO()
            image.save(buf, format="PNG")
            b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
            urls.append(f"data:image/png;base64,{b64_img}")

        return urls, n_pages

    def _call_chat_completions(self, messages: List[Dict[str, Any]]) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": {self.model},
            "messages": messages,
            "temperature": 0.0,
        }
        resp = self.http_client.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return json.dumps(data, ensure_ascii=False)

    def _call_responses(self, messages: List[Dict[str, Any]]) -> str:
        # Fallback: stuff the messages into a single text "input"
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }
        resp = self.http_client.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        # Try to read a common shape
        out = ""
        for item in data.get("output", []) or []:
            if item.get("type") == "message":
                for part in item.get("content", []) or []:
                    if part.get("type") == "output_text":
                        out += part.get("text", "")
        return out or json.dumps(data, ensure_ascii=False)

    def parse_pdf_to_markdown(
        self,
        pdf_path: str,
        *,
        prompt: str = "Parse this document to text. Output markdown",
        system: str = "Tu es un expert dans le parsing de documents PDF.",
        dpi: int = 300,
        max_pages: Optional[int] = None,  # set None to send ALL pages (may be large)
    ) -> Tuple[str, int]:
        """
        Returns (markdown_text, total_page_count).
        NOTE: Sending many pages inline can be large. By default, we send all pages;
              you can cap with max_pages for safety.
        """
        data_urls, total_pages = self._pdf_to_data_urls(pdf_path, dpi=dpi, max_pages=max_pages)
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for url in data_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        try:
            text = self._call_chat_completions(messages)
        except Exception:
            text = self._call_responses(messages)
        return text, total_pages


# ------------------------
# Reranker client
# ------------------------

class RerankerClient:
    """
    Thin wrapper around a Qwen Reranker-style /v1/score endpoint.

    Expected payload (based on your working snippet):
        POST {base_url}
        {
          "model": model_name,
          "text_1": [query_prompt],
          "text_2": [doc_prompt_1, doc_prompt_2, ...],
          "truncate_prompt_tokens": -1
        }

    We return a list of (score, label) for each document.
    If the service only returns scores, labels are derived using the
    provided threshold.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 30.0,
        instruction: Optional[str] = None,
        threshold: float = 0.9,
        http_client: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.threshold = threshold
        if http_client is None:
            http_client = _build_requests_session()
        self.http_client = http_client

        # Defaults from the user's example
        self.prefix = (
            '<|im_start|>system\n'
            'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
            'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.instruction = instruction or (
            "Given a web search query, retrieve relevant passages that answer the query"
        )

    def _make_prompts(self, query: str, docs: List[str]) -> Tuple[List[str], List[str]]:
        query_prompt = f"{self.prefix}<Instruct>: {self.instruction}\n<Query>: {query}\n"
        doc_prompts = [f"<Document>: {d}{self.suffix}" for d in docs]
        return [query_prompt], doc_prompts

    def score(self, query: str, docs: List[str]) -> List[Dict[str, Any]]:
        if not docs:
            return []
        text_1, text_2 = self._make_prompts(query, docs)
        try:
            resp = self.http_client.post(
                self.base_url,
                json={
                    "model": self.model,
                    "text_1": text_1,
                    "text_2": text_2,
                    "truncate_prompt_tokens": -1,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json() or {}
        except Exception as e:
            # Fail soft: return zeros to keep pipeline resilient
            return [
                {"score": 0.0, "label": "no", "error": f"reranker_error: {e}"}
                for _ in docs
            ]

        scores = self._extract_scores(data, len(docs))
        labels = ["yes" if s >= self.threshold else "no" for s in scores]
        return [{"score": s, "label": lbl} for s, lbl in zip(scores, labels)]

    @staticmethod
    def _extract_scores(payload: Dict[str, Any], n: int) -> List[float]:
        """Best-effort extraction across a few likely shapes."""
        # 1) {"scores": [..]}
        if isinstance(payload.get("scores"), list) and payload["scores"]:
            scores = payload["scores"]
            # If nested shape, flatten the first row
            if isinstance(scores[0], list):
                return [float(x) for x in scores[0][:n]]
            return [float(x) for x in scores[:n]]

        # 2) {"data": [{"score": ..}, ...]}
        data = payload.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict) and "score" in data[0]:
            return [float(d.get("score", 0.0)) for d in data[:n]]

        # 3) {"outputs": [{"scores": [...]}, ...]}
        outs = payload.get("outputs") or payload.get("output")
        if isinstance(outs, list) and outs:
            first = outs[0]
            if isinstance(first, dict) and isinstance(first.get("scores"), list):
                return [float(x) for x in first["scores"][:n]]

        # Fallback zeros
        return [0.0] * n


# ------------------------
# Vector store wrapper
# ------------------------

class EmbeddingStore:
    """Thin wrapper around ChromaDB with CSV upsert & vector search."""

    def __init__(self, client: Optional[chromadb.Client] = None, collection_name: str = "docs"):
        self.client = client or chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)

    @staticmethod
    def _parse_embedding(val: Any) -> np.ndarray:
        if isinstance(val, str):
            # handle cases like "0.1, 0.2, 0.3" OR "[0.1, 0.2, 0.3]"
            s = val.strip()
            if not s.startswith("["):
                s = f"[{s}]"
            return np.array(ast.literal_eval(s), dtype="float32")
        return np.array(val, dtype="float32")

    def upsert_from_csv(
        self,
        csv_path: str,
        *,
        content_col: str = "content",
        embedding_col: str = "embedding",
        metadata_col: str = "metadata",
        id_prefix: str = "",
    ) -> None:
        df = pd.read_csv(csv_path)
        if embedding_col not in df.columns:
            raise ValueError(f"CSV missing column: {embedding_col}")
        if content_col not in df.columns:
            raise ValueError(f"CSV missing column: {content_col}")
        if metadata_col not in df.columns:
            # default empty metadata
            df[metadata_col] = "{}"

        df[embedding_col] = df[embedding_col].apply(self._parse_embedding)

        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, ast.literal_eval(row.metadata)['url'] + str(i))) for i, row in df.iterrows()]
        metadatas = []
        for m in df[metadata_col].tolist():
            if isinstance(m, str):
                try:
                    metadatas.append(ast.literal_eval(m))
                except Exception:
                    metadatas.append({})

        self.collection.add(
            documents=df[content_col].tolist(),
            metadatas=metadatas,
            embeddings=df[embedding_col].tolist(),
            ids=ids,
        )

    def add_documents(
        self,
        *,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> None:
        if ids is None:
            ids = [f"doc-{i}" for i in range(len(documents))]
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids,
        )

    def query_by_embedding(self, embedding: List[float], where: Dict = None, where_document: Optional[Dict] = None, n_results: int = 3) -> Dict[str, Any]:
        return self.collection.query(query_embeddings=[embedding], where=where, where_document=where_document, n_results=n_results)

    def query_by_metadata(self, where: Dict = None, where_document: Optional[Dict] = None, limit: int = 5) -> Dict[str, Any]:
        return self.collection.get(where=where, where_document=where_document, limit=limit)

# ------------------------
# BM25 index (global, initialized once)
# ------------------------
bm25 = None
bm25_docs = None

def init_bm25(store: EmbeddingStore):
    global bm25, bm25_docs
    if bm25 is None:
        bm25_docs = store.collection.get(include=["documents", "metadatas"])
        tokenized_docs = [doc.split() for doc in bm25_docs["documents"]]
        bm25 = BM25Okapi(tokenized_docs)

# ------------------------
# Tools
# ------------------------

@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    func: Union[Callable[..., Any], Callable[..., Awaitable[Any]]]

    def as_harmony_tool(self) -> Any:
        return ToolDescription.new(self.name, self.description, parameters=self.parameters)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def harmony_descriptions(
        self,
        *,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> List[Any]:
        """
        Returns Harmony ToolDescription list.
        - include: if provided, only these tool names are exposed to the LLM
        - exclude: if provided, these tool names are hidden from the LLM

        NOTE: This does NOT affect local execution; it's only for what the model can call.
        """
        include_set = set(include or [])
        exclude_set = set(exclude or [])

        tools = []
        for name, t in self._tools.items():
            if include_set and name not in include_set:
                continue
            if exclude_set and name in exclude_set:
                continue
            tools.append(t.as_harmony_tool())
        return tools

    async def call(self, name: str, **kwargs) -> Any:
        t = self.get(name)
        if asyncio.iscoroutinefunction(t.func):
            return await t.func(**kwargs)
        return t.func(**kwargs)


# ------------------------
# Conversation wiring (Harmony)
# ------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_REASONING_RE = re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE)
_DOC_CTX_RE = re.compile(r"</?documents_de_contexte>", re.IGNORECASE)

def strip_reasoning_traces(text: str) -> str:
    """
    Best-effort removal of visible reasoning traces if the model ever emits them
    (e.g., <think>...</think>). Does nothing if those tags are absent.
    """
    if not text:
        return text
    text = _THINK_RE.sub("", text)
    text = _REASONING_RE.sub("", text)
    return text.strip()


def unwrap_documents_de_contexte(text: str) -> str:
    """Remove the <documents_de_contexte> wrapper added in add_tool_output()."""
    if not text:
        return text
    return _DOC_CTX_RE.sub("", text).strip()


def extract_output_text(responses_obj: Any) -> str:
    """
    Extract assistant text from an OpenAI Responses API-like object, compatible with vLLM.
    """
    txt = getattr(responses_obj, "output_text", None)
    if isinstance(txt, str) and txt:
        return txt

    # Fallback: try to read from responses.output[*].content[*].text
    try:
        items = getattr(responses_obj, "output", []) or []
        parts: List[str] = []
        for it in items:
            if getattr(it, "type", None) == "message":
                for part in getattr(it, "content", []) or []:
                    if getattr(part, "type", None) == "output_text":
                        parts.append(getattr(part, "text", "") or "")
        return "\n".join(p for p in parts if p).strip()
    except Exception:
        return ""
        
class ConversationManager:
    """
    Maintains Harmony messages and renders to a single prompt string.
    NEW:
      - Supports an internal rolling memory block injected into the DEVELOPER message.
      - Supports exposing only a subset of registered tools to the LLM (e.g., hide metadata_search).
      - Supports extracting the last completed turn segment for memory compression.
      - Supports stripping tool traces after each completed turn.
    """

    def __init__(
        self,
        *,
        language_system_identity: str,
        developer_prompt: str,
        tools: ToolRegistry,
        tool_descriptions: Optional[List[Any]] = None,
    ):
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

        self.system_message = (
            SystemContent.new()
            .with_reasoning_effort(ReasoningEffort.LOW)
            .with_conversation_start_date(datetime.now().strftime("%Y-%m-%d"))
        )
        self.system_message.model_identity = language_system_identity

        # Keep a stable base developer prompt, and inject memory at send-time by rebuilding the dev message.
        self._base_developer_prompt = developer_prompt
        self._memory_block: str = ""  # stored as JSON string (or empty)
        self._tool_descs = tool_descriptions if tool_descriptions is not None else tools.harmony_descriptions()

        self.developer_message = self._build_developer_content()

        self.messages: List[Message] = [
            Message.from_role_and_content(Role.SYSTEM, self.system_message),
            Message.from_role_and_content(Role.DEVELOPER, self.developer_message),
        ]

    # -----------------
    # Developer message (base prompt + rolling memory)
    # -----------------

    def _compose_developer_instructions(self) -> str:
        base = (self._base_developer_prompt or "").rstrip()

        # Memory is never user-facing. It's appended and clearly labeled.
        if self._memory_block and self._memory_block.strip():
            return (
                base
                + "\n\n"
                + "[Bloc de mémoire interne — non visible à l'utilisateur]\n"
                + "<memory>\n"
                + self._memory_block.strip()
                + "\n</memory>\n"
            )

        return base + "\n"

    def _build_developer_content(self) -> DeveloperContent:
        return (
            DeveloperContent.new()
            .with_instructions(self._compose_developer_instructions())
            .with_function_tools(self._tool_descs)
        )

    @property
    def memory_block(self) -> str:
        return self._memory_block

    def set_memory_block(self, memory_json: str) -> None:
        """
        Update rolling memory (as JSON string) and rebuild the Developer message in-place.
        """
        self._memory_block = (memory_json or "").strip()
        self.developer_message = self._build_developer_content()
        # Replace the developer message at index 1
        if len(self.messages) >= 2:
            self.messages[1] = Message.from_role_and_content(Role.DEVELOPER, self.developer_message)

    # -----------------
    # Conversation lifecycle
    # -----------------

    def reset_to_preamble(self, *, clear_memory: bool = True):
        if clear_memory:
            self._memory_block = ""
        self.developer_message = self._build_developer_content()
        self.messages = [
            Message.from_role_and_content(Role.SYSTEM, self.system_message),
            Message.from_role_and_content(Role.DEVELOPER, self.developer_message),
        ]

    def add_user(self, text: str):
        self.messages.append(Message.from_role_and_content(Role.USER, text))

    def add_assistant_text(self, text: str):
        self.messages.append(Message.from_role_and_content(Role.ASSISTANT, text))

    def add_tool_output(self, tool_name: str, output_payload: str):
        # Mirrors your notebook: tool messages use Author.new(Role.TOOL, f"functions.{name}")
        tool_author = Author.new(Role.TOOL, f"functions.{tool_name}")
        wrapped = f"<documents_de_contexte>{output_payload}</documents_de_contexte>"
        self.messages.append(
            Message.from_author_and_content(tool_author, wrapped).with_channel("commentary")
        )

    # -----------------
    # Prompt rendering
    # -----------------

    def render_prompt(self, tokenizer: TokenizerClient) -> str:
        convo = Conversation.from_messages(self.messages)
        tokens = self.encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        prompt = tokenizer.detokenize(tokens)["prompt"]
        return prompt

    def count_tokens(self, tokenizer: TokenizerClient) -> int:
        convo = Conversation.from_messages(self.messages)
        tokens = self.encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        return len(tokens)

    # -----------------
    # NEW: Turn extraction + trace stripping
    # -----------------

    def extract_last_turn_segment(self) -> Dict[str, Any]:
        """
        Returns the most recent completed turn as a dict:
          {
            "user": <str>,
            "assistant": <str>,
            "tools": [ {"name": <tool_name>, "content": <raw_content>} , ... ]
          }

        The segment includes any tool messages that occurred between the previous assistant reply
        and the latest assistant reply (including pdf_upsert messages inserted before the user message).
        """
        # Find latest assistant message (end of current turn)
        last_assistant_idx = None
        for i in range(len(self.messages) - 1, -1, -1):
            if getattr(self.messages[i], "role", None) == Role.ASSISTANT:
                last_assistant_idx = i
                break
        if last_assistant_idx is None:
            return {"user": "", "assistant": "", "tools": []}

        # Find previous assistant message (end of previous turn)
        prev_assistant_idx = None
        for i in range(last_assistant_idx - 1, -1, -1):
            if getattr(self.messages[i], "role", None) == Role.ASSISTANT:
                prev_assistant_idx = i
                break

        start_idx = (prev_assistant_idx + 1) if prev_assistant_idx is not None else 2  # after sys+dev
        seg = self.messages[start_idx : last_assistant_idx + 1]

        user_text = ""
        assistant_text = ""
        tools_out: List[Dict[str, str]] = []

        for m in seg:
            role = getattr(m, "role", None)
            if role == Role.USER:
                user_text = getattr(m, "content", "") or ""
            elif role == Role.ASSISTANT:
                assistant_text = getattr(m, "content", "") or ""
            elif role == Role.TOOL:
                author = getattr(m, "author", None)
                author_name = getattr(author, "name", "") if author is not None else ""
                tool_name = author_name.split("functions.", 1)[-1] if "functions." in author_name else author_name
                tools_out.append({"name": tool_name, "content": getattr(m, "content", "") or ""})

        return {"user": user_text, "assistant": assistant_text, "tools": tools_out}

    def strip_all_tool_traces(self) -> None:
        """
        Removes ALL Role.TOOL messages from history to keep context clean between turns.
        Call this AFTER you've updated rolling memory.
        """
        if len(self.messages) <= 2:
            return
        head = self.messages[:2]
        tail = [m for m in self.messages[2:] if getattr(m, "role", None) != Role.TOOL]
        self.messages = head + tail


# ------------------------
# Tool runner & inference loop
# ------------------------

class ToolRunner:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.logger = logging.getLogger("ToolRunner")

    async def execute_function_calls(self, assistant_output: Any) -> List[Dict[str, Any]]:
        """
        Parse `assistant_output` (Responses API object) to find function calls,
        execute them, and return a list of {name, call_id, arguments, output} dicts.
        """
        calls: List[Dict[str, Any]] = []

        try:
            items = getattr(assistant_output, "output", [])
        except Exception:
            items = []

        for item in items:
            if getattr(item, "type", None) == "function_call":
                name = getattr(item, "name", "")
                args_str = getattr(item, "arguments", "{}")
                call_id = getattr(item, "call_id", "")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
                except Exception:
                    args = {}

                self.logger.debug("Calling tool %s with args %s", name, args)
                try:
                    result = await self.registry.call(name, **args)
                except Exception as e:
                    result = {"error": str(e)}

                # Ensure string payload (your notebook wraps str(func_output))
                try:
                    payload = json.dumps(result, ensure_ascii=False)
                except Exception:
                    payload = str(result)

                calls.append(
                    {
                        "name": name,
                        "call_id": call_id,
                        "arguments": args,
                        "output": payload,
                    }
                )
        return calls


async def run_llm_round(
    *,
    conv: ConversationManager,
    llm: LLMClient,
    tokenizer: TokenizerClient,
    tools: ToolRunner,
    temperature: float = 0.0,
    max_tool_loops: int = 3,
    debug_history: bool = False,
) -> Tuple[str, int]:
    """
    Runs a single conversational turn with automatic tool execution.
    NEW behavior:
      - Allows up to `max_tool_loops` tool iterations.
      - If tool loop limit is hit, forces a final answer (tool_choice="none") instead of returning a guard.
      - Strips visible reasoning traces (e.g., <think>...</think>) from the final answer if present.
    Returns: (final_assistant_text, prompt_token_count)
    """
    # Up to max_tool_loops tool cycles. Each cycle: ask model -> execute tools -> append tool outputs -> repeat.
    for _ in range(max_tool_loops + 1):
        prompt = conv.render_prompt(tokenizer)

        if debug_history:
            print("\n\033[90m========== LLM PROMPT (rendered) ==========\033[0m")
            print(prompt)
            print("\033[90m========== END PROMPT ==========\033[0m\n")
        
            print("\033[90m========== HARMONY MESSAGES ==========\033[0m")
            for i, m in enumerate(conv.messages):
                role = getattr(m, "role", None)
                author = getattr(m, "author", None)
                author_name = getattr(author, "name", "") if author is not None else ""
                content = getattr(m, "content", "")
                print(f"\n--- [{i}] role={role} author={author_name}")
                print(content)
            print("\033[90m========== END HARMONY MESSAGES ==========\033[0m\n")
        
        assistant = await llm.respond(prompt, temperature=temperature, tool_choice="auto")

        # Execute requested function calls (if any)
        executed = await tools.execute_function_calls(assistant)
        if executed:
            for fc in executed:
                conv.add_tool_output(fc["name"], fc["output"])
            continue

        # No tool calls => finalize
        final_text = extract_output_text(assistant)
        final_text = strip_reasoning_traces(final_text)

        conv.add_assistant_text(final_text)
        n_tokens = conv.count_tokens(tokenizer)
        return final_text, n_tokens

    # If we reached here, the model kept calling tools too many times.
    # Force a final answer without tools.
    prompt = conv.render_prompt(tokenizer)
    assistant = await llm.respond(prompt, temperature=temperature, tool_choice="auto")

    final_text = extract_output_text(assistant)
    final_text = strip_reasoning_traces(final_text)

    conv.add_assistant_text(final_text)
    n_tokens = conv.count_tokens(tokenizer)
    return final_text, n_tokens

class MemoryCompressor:
    """
    Rolling memory compressor that:
      - Extracts doc refs + PDF uploads deterministically from tool outputs
      - Uses an extra LLM call (no tools) to update concise key_facts/decisions/open_questions
      - Returns a JSON string that is injected into the developer message <memory> block

    IMPORTANT: Memory is internal only. Keep it short and factual.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        max_doc_refs: int = 40,
        max_key_facts: int = 20,
        max_decisions: int = 15,
        max_open_questions: int = 8,
    ):
        self.llm = llm
        self.max_doc_refs = max_doc_refs
        self.max_key_facts = max_key_facts
        self.max_decisions = max_decisions
        self.max_open_questions = max_open_questions

    @staticmethod
    def _safe_json_load(s: str) -> Optional[Dict[str, Any]]:
        if not s or not isinstance(s, str):
            return None
        try:
            return json.loads(s)
        except Exception:
            return None

    @staticmethod
    def _dedupe_keep_recent(items: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
        seen = set()
        out_rev: List[Dict[str, Any]] = []
        for it in reversed(items):
            k = (it or {}).get(key)
            if not k or k in seen:
                continue
            seen.add(k)
            out_rev.append(it)
        return list(reversed(out_rev))

    def _extract_sources_from_tools(self, tool_msgs: List[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Parses tool message payloads to extract:
          - doc_refs: [{doc_id, source, title, url, tags?}, ...]
          - uploaded_pdfs: [{source, upserted_at, page_count, chunks, ids_range?}, ...]
        """
        doc_refs: List[Dict[str, Any]] = []
        uploaded_pdfs: List[Dict[str, Any]] = []

        for tm in tool_msgs or []:
            name = (tm.get("name") or "").strip()
            raw = unwrap_documents_de_contexte(tm.get("content") or "")
            payload = self._safe_json_load(raw)

            if not isinstance(payload, dict):
                continue

            # semantic_search / metadata_search return doc_ids + doc_metadatas
            if name in {"semantic_search", "metadata_search"}:
                ids = payload.get("doc_ids") or []
                metas = payload.get("doc_metadatas") or []
                if isinstance(ids, list) and isinstance(metas, list):
                    for _id, meta in zip(ids, metas):
                        if not isinstance(meta, dict):
                            meta = {}
                        doc_refs.append(
                            {
                                "doc_id": _id,
                                "source": meta.get("source"),
                                "title": meta.get("title"),
                                "url": meta.get("url"),
                                "tags": meta.get("tags"),
                            }
                        )

            # pdf_upsert trace
            if name == "pdf_upsert" and payload.get("event") == "pdf_upsert":
                uploaded_pdfs.append(
                    {
                        "source": payload.get("source"),
                        "upserted_at": payload.get("upserted_at"),
                        "page_count": payload.get("page_count"),
                        "chunks": payload.get("chunks"),
                        "ids_range": payload.get("ids_range"),
                    }
                )

        # de-dupe doc refs by doc_id, keep most recent
        doc_refs = self._dedupe_keep_recent(doc_refs, key="doc_id")
        uploaded_pdfs = self._dedupe_keep_recent(uploaded_pdfs, key="upserted_at")

        return doc_refs, uploaded_pdfs

    async def _llm_update_lists(
        self,
        *,
        prior_memory: Dict[str, Any],
        user_text: str,
        assistant_text: str,
        doc_refs: List[Dict[str, Any]],
        uploaded_pdfs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Uses a tool-less LLM call to update:
          - key_facts
          - decisions
          - open_questions
        Returns dict with those keys.
        """
        prior_key_facts = prior_memory.get("key_facts") or []
        prior_decisions = prior_memory.get("decisions") or []
        prior_open = prior_memory.get("open_questions") or []

        # Keep tool outputs out of this prompt (too big). Provide doc refs metadata only.
        prompt = f"""
Tu es un module interne de compression de mémoire pour un chatbot AVS/AI.
Règles:
- N'invente rien. Utilise UNIQUEMENT les informations fournies ci-dessous.
- Pas de données personnelles.
- Sois concis. Évite les doublons.
- Mets à jour 3 listes: key_facts, decisions, open_questions.

Entrées:
1) Mémoire précédente (listes):
- key_facts: {json.dumps(prior_key_facts, ensure_ascii=False)}
- decisions: {json.dumps(prior_decisions, ensure_ascii=False)}
- open_questions: {json.dumps(prior_open, ensure_ascii=False)}

2) Dernier tour:
- message utilisateur: {user_text}
- réponse assistant: {assistant_text}

3) Références docs récupérées (métadonnées uniquement, peut être vide):
{json.dumps(doc_refs, ensure_ascii=False)}

4) PDFs uploadés récemment (peut être vide):
{json.dumps(uploaded_pdfs, ensure_ascii=False)}

Sortie:
Retourne UNIQUEMENT un objet JSON valide EXACTEMENT avec les clés:
{{"key_facts": [...], "decisions": [...], "open_questions": [...]}}
""".strip()

        resp = await self.llm.respond(prompt, temperature=0.0, tool_choice="auto")
        out = extract_output_text(resp).strip()

        updated = self._safe_json_load(out)
        if not isinstance(updated, dict):
            # Fail-soft: keep previous lists
            return {
                "key_facts": prior_key_facts,
                "decisions": prior_decisions,
                "open_questions": prior_open,
            }

        return {
            "key_facts": updated.get("key_facts", prior_key_facts),
            "decisions": updated.get("decisions", prior_decisions),
            "open_questions": updated.get("open_questions", prior_open),
        }

    async def update_memory_json(
        self,
        *,
        prior_memory_json: str,
        turn_user: str,
        turn_assistant: str,
        turn_tools: List[Dict[str, str]],
    ) -> str:
        """
        Main entry: returns updated memory as JSON string.
        """
        prior = self._safe_json_load(prior_memory_json) if prior_memory_json else None
        prior = prior if isinstance(prior, dict) else {}

        doc_refs, uploaded_pdfs = self._extract_sources_from_tools(turn_tools)

        # Merge doc refs & pdf uploads with prior, keeping recent and capped
        merged_doc_refs = (prior.get("doc_refs") or []) + doc_refs
        if isinstance(merged_doc_refs, list):
            merged_doc_refs = self._dedupe_keep_recent(merged_doc_refs, key="doc_id")
        else:
            merged_doc_refs = doc_refs

        merged_uploaded = (prior.get("uploaded_pdfs") or []) + uploaded_pdfs
        if isinstance(merged_uploaded, list):
            merged_uploaded = self._dedupe_keep_recent(merged_uploaded, key="upserted_at")
        else:
            merged_uploaded = uploaded_pdfs

        # LLM updates for key_facts/decisions/open_questions
        lists = await self._llm_update_lists(
            prior_memory=prior,
            user_text=turn_user,
            assistant_text=turn_assistant,
            doc_refs=doc_refs[-10:],         # give only recent refs to keep prompt small
            uploaded_pdfs=uploaded_pdfs[-5:],
        )

        # Final memory object (cap sizes)
        memory = {
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "key_facts": (lists.get("key_facts") or [])[: self.max_key_facts],
            "decisions": (lists.get("decisions") or [])[: self.max_decisions],
            "open_questions": (lists.get("open_questions") or [])[: self.max_open_questions],
            "doc_refs": merged_doc_refs[-self.max_doc_refs :],
            "uploaded_pdfs": merged_uploaded[-10:],  # small cap; adjust if needed
        }

        return json.dumps(memory, ensure_ascii=False, indent=2)
# ------------------------
# Ready-made RAG search tool (with optional reranking/validation)
# ------------------------
class RAGTool:
    _bm25 = None
    _bm25_docs = None
    
    def __init__(
        self,
        store: EmbeddingStore,
        *,
        reranker: Optional[RerankerClient] = None,
        validation_mode: str = "off",
        verbose: bool = False,
    ):
        self.store = store
        self.reranker = reranker
        self.validation_mode = validation_mode
        self.verbose = verbose
        
        if RAGTool._bm25 is None:
            self._refresh_bm25()

    # -----------------
    # Protected helpers
    # -----------------

    def _refresh_bm25(self):
        docs = self.store.collection.get(include=["documents", "metadatas"])
        tokenized_docs = [doc.split() for doc in docs.get("documents", [])]
        RAGTool._bm25_docs = docs
        RAGTool._bm25 = BM25Okapi(tokenized_docs) if tokenized_docs else None
        if self.verbose:
            print(f"[bm25] Refreshed index with {len(tokenized_docs)} docs")

    async def _apply_bm25(self, query: str, top_k: int = 5):
        """Return BM25 results (id, content, metadata, score)."""
        if RAGTool._bm25 is None:
            return []

        scores = RAGTool._bm25.get_scores(query.split())
        docs = RAGTool._bm25_docs
        results = [
            {"id": docs["ids"][i], "metadata": docs["metadatas"][i],
             "score": scores[i], "content": docs["documents"][i]}
            for i in range(len(scores))
        ]
        results.sort(key=lambda x: x["score"], reverse=True)

        if self.verbose:
            print("\n>>> [BM25 results]\n", "\n".join([f'id: {r["id"]}\nurl: {r["metadata"].get("url")}\ntitle: {r["metadata"].get("title")}\nsource: {r["metadata"].get("source")}\n' for r in results[:top_k]]))

        return results[:top_k]

    def _apply_reranker(self, query: str, docs: List[str], ids: List[str], metas: List[Dict[str, Any]]):
        """Apply reranker to docs, return merged results with scores/labels."""
        if not (self.reranker and docs and self.validation_mode in {"rerank", "filter"}):
            return ids, docs, metas, [], []

        print("\n\33[35m>>> Reranking documents")
        scored = self.reranker.score(query, docs)

        if self.verbose:
            score_info = [(m.get("url") or m.get("title"), m.get("source"), s["label"], s["score"]) for m, s in zip(metas, scored)]
            out = "\n".join(f"url:{url}\nsource:{source}\nlabel:{label}\nscore:{score:.3f}\n" for url, source, label, score in score_info)
            print("\n>>> [Reranker Scores]\n", out)

        merged = [
            {
                "id": i,
                "content": d,
                "metadata": metas[idx],
                "reranker": scored[idx],
            }
            for idx, (i, d) in enumerate(zip(ids, docs))
        ]
        merged.sort(key=lambda x: x["reranker"]["score"], reverse=True)

        if self.validation_mode == "filter":
            merged = [m for m in merged if m["reranker"]["label"] == "yes"]

        ids = [m["id"] for m in merged]
        docs = [m["content"] for m in merged]
        metas = [m["metadata"] for m in merged]
        scores = [m["reranker"]["score"] for m in merged]
        labels = [m["reranker"]["label"] for m in merged]
        return ids, docs, metas, scores, labels


class SemanticSearchTool(RAGTool):
    
    def build(self, embedder: EmbeddingClient, n_results: int = 3) -> Tool:
        async def _search(query: str, where: Optional[Dict] = None, where_document: Optional[Dict] = None):
            print("\n\33[35m>>> Calling `semantic_search` function.")

            vec = await embedder.embed(query)

            print(f"\n>>> [Semantic Search Query Params] {query} | {where} | {where_document}")
            results = self.store.query_by_embedding(vec, where=where, where_document=where_document, n_results=n_results)

            if self.verbose:
                print("\n>>> [Semantic Search results]\n", "\n".join([f'***DOC {i}***\nid: {_id}\nurl: {m.get("url")}\ntitle: {m.get("title")}\nsource: {m.get("source")}\n' for i, (_id, m) in enumerate(zip(results["ids"][0], results["metadatas"][0]))]))

            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]

            bm25_results = await self._apply_bm25(query, top_k=n_results)
                
            
            for hit in bm25_results:
                ids.append(hit["id"])
                docs.append(hit["content"])
                metas.append(hit["metadata"])
                dists.append(float(hit["score"]))

            # drop duplicates between semantic + bm25 (by doc id, keep first occurrence)
            seen = set()
            ids, docs, metas, dists = map(list, zip(*[(i,d,m,s) for i,d,m,s in zip(ids,docs,metas,dists) if not (i in seen or seen.add(i))]))

            ids, docs, metas, scores, labels = self._apply_reranker(query, docs, ids, metas)

            return {
                "doc_ids": ids,
                "doc_metadatas": metas,
                "content": docs,
                "reranker_scores": scores,
                "reranker_labels": labels,
                "validation_mode": self.validation_mode,
            }
    
        return Tool(
            name="semantic_search",
            description=(
                "Gets up-to-date documents about the Swiss 1st-pillar social insurance (AVS/AI) "
                "from a vector DB (semantic search) to answer the user query. Optionally reranks/filters results "
                "using an external reranker (qwen-reranker). Can also take `where` and `where_document` args to filter on metadata."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user query to perform semantic search upon.",
                    },
                    "where": {
                        "type": "object",
                        "description": "Optional metadata filters for restricting search (same as Chroma's `where`)."
                        "Supports $eq, $gt, $lt, $and, $or, $in, $nin operators."
                        "Ex: { 'language': 'fr' }"
                        "Ex: { 'subsection': 'art_24' }"
                        "Ex: { '$and': [ {'page': {'$gte': 5 }}, {'page': {'$lte': 10 }}, ] }"
                        "Ex: { '$and': [ {'last_modification': {'$gte': '01.01.2025' }}, {'last_modification': {'$lte': '08.09.2025' }}, ] }"
                        "Ex: { 'or': [ {'subtopics': 'avs'}, {'subtopics': 'ai'}, ] }"
                        "Ex: { 'source': {'$in': ['ahv_iv_mementos', 'lavs', 'csc_afac']} }"
                        "Ex: { 'tags': {'$nin': ['rente_ai', 'indépendants', 'assujetissement_avs']} }"
                    },
                    "where_document": {
                        "type": "object",
                        "description": (
                            "Optional metadata filters for full-text search (same as Chroma's `where_document`)."
                            "Supports $contains, $not_contains, $regex, $not_regex, $and, $or operators."
                            "Ex: { '$and': [ {'$contains': 'search_string_1'}, {'$regex': '[a-z]+'}, ] }"
                            "Ex: { '$or': [ {'$contains': 'search_string_1'}, {'$not_contains': 'search_string_2'}, ] }"
                        ),
                    },
                    "required": ["query"],
                },
            },
            func=_search,
        )

class MetadataSearchTool(RAGTool):
    
    def build(self, limit: int = 5) -> Tool:
        async def _search(query: str, where: Optional[Dict] = None, where_document: Optional[Dict] = None):
            print("\n\33[35m>>> Calling `metadata_search` function.")

            print(f"\n>>> [Metadata Search Query Params] {where} | {where_document}")
            results = self.store.query_by_metadata(where=where, where_document=where_document, limit=limit)

            if self.verbose:
                print("\n>>> [Metadata Search results]", [(_id, m.get("url") or m.get("title"), m.get("source")) for _id, m in zip(results["ids"], results["metadatas"])])

            """ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]"""

            ids = results["ids"]
            docs = results["documents"]
            metas = results["metadatas"]

            """bm25_results = await self._apply_bm25(query, top_k=limit)
            
            for hit in bm25_results:
                ids.append(hit["id"])
                docs.append(hit["content"])
                metas.append(hit["metadata"])

            

            ids, docs, metas, scores, labels = self._apply_reranker(query, docs, ids, metas)"""

            print(query)
            
            return {
                "doc_ids": ids,
                "doc_metadatas": metas,
                "content": docs,
                "reranker_scores": [],
                "reranker_labels": [],
                "validation_mode": self.validation_mode,
            }

        return Tool(
            name="metadata_search",
            description="Performs metadata/full-text search with optional reranking/filtering.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user query to perform BM25 search upon.",
                    },
                    "where": {
                        "type": "object",
                        "description": "Optional metadata filters for restricting search (same as Chroma's `where`)."
                        "Supports $eq, $gt, $lt, $and, $or, $in, $nin operators."
                        "Ex: { 'language': 'fr' }"
                        "Ex: {'subsection': 'art_24' }"
                        "Ex: { '$and': [ {'page': {'$gte': 5 }}, {'page': {'$lte': 10 }}, ] }"
                        "Ex: { '$and': [ {'last_modification': {'$gte': '01.01.2025' }}, {'last_modification': {'$lte': '08.09.2025' }}, ] }"
                        "Ex: { 'or': [ {'subtopics': 'avs'}, {'subtopics': 'ai'}, ] }"
                        "Ex: { 'source': {'$in': ['ahv_iv_mementos', 'lavs', 'csc_afac']} }"
                        "Ex: { 'tags': {'$nin': ['rente_ai', 'indépendants', 'assujetissement_avs']} }"
                    },
                    "where_document": {
                        "type": "object",
                        "description": (
                            "Optional metadata filters for full-text search (same as Chroma's `where_document`)."
                            "Supports $contains, $not_contains, $regex, $not_regex, $and, $or operators."
                            "Ex: { '$and': [ {'$contains': 'search_string_1'}, {'$regex': '[a-z]+'}, ] }"
                            "Ex: { '$or': [ {'$contains': 'search_string_1'}, {'$not_contains': 'search_string_2'}, ] }"
                        ),
                },
                "required": ["where"],
                },
            },
            func=_search,
        )

# ------------------------
# Small helpers
# ------------------------

ZAS_SYSTEM_IDENTITY_FR = """Vous êtes le ZIA, un assistant consciencieux et engagé qui fournit des réponses détaillées et précises aux questions du public sur les assurances sociales en Suisse.
    Toutes les demandes concernent obligatoirement les assurances sociales 1er pilier (AVS/AI), la Centrale de Compensation CdC,
    la Caisse Fédérale de Compensation (CFC), l'Assurance Facultative (AF), l'Office AI des suisses de l'Etranger (OAIE),
    la Caisse Suisse de Compensation (CSC), le cadre légal (fedlex), les Ressources Humaines (RH) ainsi que les mémentos et directives internes à la CdC/CFC."""


def read_developer_prompt(path: str, *, response_format: str = "Concise") -> str:
    with open(path, "r", encoding="utf-8") as f:
        template = f.read()
    # Your notebook uses developer_message.format(response_format="Concise")
    try:
        return template.format(response_format=response_format)
    except Exception:
        return template


def _chunk_markdown(text: str, max_chars: int = 1500) -> List[str]:
    """
    Extremely simple chunker: splits on blank lines, then packs paragraphs up to max_chars.
    Keeps it dependency-less and deterministic.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    size = 0
    for p in paras:
        if size + len(p) + 2 > max_chars and buf:
            chunks.append("\n\n".join(buf))
            buf, size = [p], len(p)
        else:
            buf.append(p)
            size += len(p) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


async def upsert_pdf_into_store(
    *,
    pdf_path: str,
    embedder: EmbeddingClient,
    store: EmbeddingStore,
    vlm: VLMParserClient,
    collection_name: str,
    max_pages: Optional[int] = None,
) -> Dict[str, Any]:
    """
    End-to-end: parse PDF to markdown, chunk, embed, upsert into Chroma.
    Returns a trace dict to insert into conversation history.
    """
    print(f"\033[96m[pdf] parsing -> {pdf_path}\033[0m")
    md_text, total_pages = vlm.parse_pdf_to_markdown(pdf_path, max_pages=max_pages)
    print(f"\033[96m[pdf] parsed {len(md_text)} chars from {total_pages} pages\033[0m")

    chunks = _chunk_markdown(md_text, max_chars=1500)
    print(f"\033[96m[pdf] chunked into {len(chunks)} segments for embedding\033[0m")

    embeddings = await embedder.embed_batch(chunks)

    ts = datetime.utcnow().isoformat() + "Z"
    pdf_name = os.path.basename(pdf_path)

    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []
    docs: List[str] = []

    # TO DO: infer metadata with LLM (subtopics/tags, etc.) -> Add to trace !!!
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        meta = {
            "pdf_name": pdf_name,
            "upserted_at": ts,
            "page_count": total_pages,
            "chunk_idx": i,
            "source": pdf_name,
        }
        metadatas.append(meta)
        _id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{pdf_name}:{ts}:{i}"))
        ids.append(_id)
        docs.append(chunk)

    store.add_documents(documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids)

    trace = {
        "event": "pdf_upsert",
        #"collection": collection_name,
        #"pdf_name": pdf_name,
        "ids": ids,
        "source": pdf_name,
        "upserted_at": ts,
        "page_count": total_pages,
        "chunks": len(chunks),
        "ids_range": [ids[0], ids[-1]] if ids else [],
    }
    
    return trace
