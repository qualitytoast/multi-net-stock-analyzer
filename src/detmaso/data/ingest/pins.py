"""A dataset's pins (the exact source version: git commits, sha256 hashes, download URL and time) are written once,
in a ``KEY="value"`` file under ``scripts/``. The download script sources that file; the converter or check reads it
with ``read_pins``. Only plain literal values are accepted, so the file means the same to both readers."""

from __future__ import annotations

import re
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts"
_PIN_LINE = re.compile(r'([A-Z][A-Z0-9_]*)="([^"$`\\]+)"')


def read_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        found = _PIN_LINE.fullmatch(line)
        if found is None or found[1] in pins:
            raise ValueError(f'{path}:{number}: expected one KEY="value" line per pin, a plain value, each key once')
        pins[found[1]] = found[2]
    if not pins:
        raise ValueError(f"no pins in {path}")
    return pins
