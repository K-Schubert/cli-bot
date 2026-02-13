from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from openai_harmony import Role

from utils.utils import (
    ConversationManager,
    ToolRegistry,
    _message_content_text,
    _message_role,
)


def _build_conversation_manager() -> ConversationManager:
    registry = ToolRegistry()
    return ConversationManager(
        language_system_identity="test",
        developer_prompt="developer",
        tools=registry,
        tool_descriptions=[],
    )


def test_truncate_to_recent_turns_limits_history():
    conv = _build_conversation_manager()

    for idx in range(5):
        conv.add_user(f"user-{idx}")
        conv.add_assistant_text(f"assistant-{idx}")

    conv.truncate_to_recent_turns(3)

    user_messages = [
        _message_content_text(msg)
        for msg in conv.messages
        if _message_role(msg) == Role.USER
    ]
    assistant_messages = [
        _message_content_text(msg)
        for msg in conv.messages
        if _message_role(msg) == Role.ASSISTANT
    ]

    assert user_messages == ["user-2", "user-3", "user-4"]
    assert assistant_messages == ["assistant-2", "assistant-3", "assistant-4"]
    assert _message_role(conv.messages[0]) == Role.SYSTEM
    assert _message_role(conv.messages[1]) == Role.DEVELOPER


def test_truncate_to_recent_turns_keeps_adjacent_tool_messages():
    conv = _build_conversation_manager()

    conv.add_user("user-0")
    conv.add_assistant_text("assistant-0")

    conv.add_tool_output("pdf_upsert", "{\"event\": \"pdf_upsert\"}")
    conv.add_user("user-1")

    conv.truncate_to_recent_turns(1)

    roles = [_message_role(msg) for msg in conv.messages]

    assert roles[:2] == [Role.SYSTEM, Role.DEVELOPER]
    assert roles[2] == Role.TOOL
    assert roles[3] == Role.USER
    assert _message_content_text(conv.messages[3]) == "user-1"