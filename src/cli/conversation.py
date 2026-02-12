from __future__ import annotations

from utils.utils import (
    ConversationManager,
    MemoryCompressor,
    ZAS_SYSTEM_IDENTITY_FR,
    read_developer_prompt,
)


def build_conversation_manager(
    *,
    developer_prompt_path: str,
    tool_registry,
    tool_descriptions,
) -> ConversationManager:
    developer_prompt = read_developer_prompt(developer_prompt_path, response_format="Concise")
    return ConversationManager(
        language_system_identity=ZAS_SYSTEM_IDENTITY_FR,
        developer_prompt=developer_prompt,
        tools=tool_registry,
        tool_descriptions=tool_descriptions,
    )


def build_memory_compressor(llm_client) -> MemoryCompressor:
    return MemoryCompressor(llm=llm_client)
