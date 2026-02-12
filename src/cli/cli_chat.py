# cli_chat5.py (updated)
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.append(_os.path.dirname(__file__))

try:  # pragma: no cover - optional dependency for sqlite backend
    __import__("pysqlite3")
    import sys as _sys2

    _sys2.modules["sqlite3"] = _sys2.modules.pop("pysqlite3")
except ImportError:
    pass

from .main import main


if __name__ == "__main__":
    main()
