"""
The agent progressively reads files to uncover a chain of "secrets" across
nested levels of folders. As the agent moves to the
next level, earlier directory listings and file views are no longer relevant
and can be redacted to reduce context.

- Relevance condenser: LLMRelevanceCondenser applies recorded directives
  inline (observations are replaced with short redaction messages while
  preserving tool-call pairing).
- Temporary workspace: created for the run and automatically cleaned up. Each
  level contains several misleading files and exactly one file whose contents
  reveal the next secret.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
import uuid

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    LLMConvertibleEvent,
)
from openhands.sdk.context.condenser import LLMRelevanceCondenser
from openhands.sdk.tool import Tool, register_tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.relevance_condenser.definition import (
    LLMRelevanceCondenserTool,
)


def _build_secret_workspace(root: Path) -> None:
    """Create a multi-level 'secret hunt' workspace under `root`.

    Semantics:
    - Root contains one useful note plus 5 dead ends.
    - Each subsequent level has 1 correct file whose CONTENTS reveal the next
      secret and 5 misleading files. Filenames intentionally contain no clues.
    """

    # Root level: create directories
    lvl1 = root / "level1"
    lvl2 = root / "level2"
    lvl3 = root / "level3"
    for d in (lvl1, lvl2, lvl3):
        d.mkdir(parents=True, exist_ok=True)

    # Root files: one starting note + 5 dead ends
    (root / "notes.txt").write_text(
        "At the next level, filenames provide no hints.\n"
        "Open and read file contents to find the hidden instruction.\n"
        "Exactly one file in each level contains a line revealing the next secret.\n"
    )
    for i in range(1, 6):
        (root / f"dead_end_{i}.txt").write_text(
            "This is a dead end. Keep exploring other files.\n"
        )

    lvl1_files = [
        "alpha.txt",
        "beta.txt",
        "gamma.txt",
        "delta.txt",
        "epsilon.txt",
        "zeta.txt",
    ]
    (lvl1 / lvl1_files[2]).write_text(
        "NEXT SECRET (for level 2): 12\n"
    )
    for fname in [f for i, f in enumerate(lvl1_files) if i != 2]:
        (lvl1 / fname).write_text(
            "This is not helpful. Try another file in this folder.\n"
        )

    lvl2_files = [
        "ornithology.txt",
        "herpetology.txt",
        "ichthyology.txt",
        "entomology.txt",
        "mammalogy.txt",
        "botany.txt",
    ]
    (lvl2 / lvl2_files[4]).write_text(
        "NEXT SECRET (for level 3): 21\n"
    )
    for fname in [f for i, f in enumerate(lvl2_files) if i != 4]:
        (lvl2 / fname).write_text(
            "This file contains field notes with no relevant secret.\n"
        )

    lvl3_files = [
        "red.txt",
        "blue.txt",
        "green.txt",
        "yellow.txt",
        "orange.txt",
        "purple.txt",
    ]
    (lvl3 / lvl3_files[1]).write_text("FINAL SECRET: 42\n")
    for fname in [f for i, f in enumerate(lvl3_files) if i != 1]:
        (lvl3 / fname).write_text(
            "This is not the file you are looking for.\n"
        )


def main() -> None:
    api_key = os.getenv("LLM_API_KEY")
    # Vertex AI ... oh gosh
    # assert api_key is not None, "LLM_API_KEY environment variable is not set."
    
    model = os.getenv("LLM_MODEL", "LLM_MODEL environment variable is not set.")
    base_url = os.getenv("LLM_BASE_URL")
    llm = LLM(
        usage_id="agent",
        model=model,
        base_url=base_url,
        api_key=SecretStr(api_key),
        native_tool_calling=True,
    )

    # Register tools
    register_tool("FileEditorTool", FileEditorTool)
    register_tool("relevance_condenser", LLMRelevanceCondenserTool)

    tools = [
        Tool(name="FileEditorTool"),
        Tool(name="relevance_condenser"),
    ]

    # Use LLMRelevanceCondenser to apply directives in-place
    condenser = LLMRelevanceCondenser()

    # Temp workspace lifecycle bound to script
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        _build_secret_workspace(root)

        # Agent with condenser and restricted tools (by instruction)
        agent = Agent(
            llm=llm,
            tools=tools,
            condenser=condenser,
            system_prompt_filename="system_prompt_short.j2",
        )

        llm_messages: list = []  # collect raw LLM-visible messages

        def conversation_callback(event):
            if isinstance(event, LLMConvertibleEvent):
                llm_messages.append(event.to_llm_message())

        # Strong guidance: only view/list, use absolute paths, follow the chain, and use the condenser tool yourself
        instruction = (
            "You are in a secret hunt inside a temporary workspace.\n"
            f"Workspace root: {root}\n\n"
            "Rules:\n"
            "- Use only the 'FileEditorTool' with the 'view' command to list directories and view files.\n"
            " via its stated usage: str_replace_editor: If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep"
            "- Always use absolute paths under the workspace root.\n"
            "- Filenames contain no clues. At each level, you must OPEN files and READ their contents to find the next secret.\n"
            "- Exactly one file per level contains either 'NEXT SECRET: <number>' or 'FINAL SECRET: 42'. Others are misleading.\n"
            "- When you progress to a new level, proactively invoke tool 'relevance_condenser' to mask earlier FileEditorTool observations that are no longer relevant.\n"
            "- Explore all levels until you find the final secret.\n"
        )

        env_cid = os.getenv("CONVERSATION_ID")
        conversation_id = None
        if env_cid:
            try:
                conversation_id = uuid.UUID(env_cid)
            except Exception:
                conversation_id = None
        conversation = Conversation(
            agent=agent,
            callbacks=[conversation_callback],
            persistence_dir="./.conversations",
            workspace=str(root),
            conversation_id=conversation_id,
        )

        print("Start/ continuing secret hunt with relevance condenser...")

        conversation.send_message(instruction)
        conversation.run()
        for i, message in enumerate(llm_messages[:8]):
            try:
                preview = str(message)[:200]
            except Exception:
                preview = "<unprintable message>"
            print(f"Message {i}: {preview}")

        print(
            "\nIf the LLM invoked 'relevance_condenser', the condenser will replace prior"
            " observations with short redaction notes while preserving tool-call"
            " pairing on subsequent steps."
        )


if __name__ == "__main__":
    main()
