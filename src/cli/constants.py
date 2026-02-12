from __future__ import annotations

import re

PDF_CMD_PATTERN = re.compile(r"/pdf\s+(.+?)(?=$|(?=\s/)|$)")
INTRO_BANNER = (
    "\nZIA — CLI (type /exit to quit, /reset to clear, /pdf <path> to upload PDF)"
)
USER_PROMPT = "\n\033[1;92muser> \033[0m"
ASSISTANT_PROMPT = "\n\033[1;94massistant> \033[0m"
