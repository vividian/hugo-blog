#!/usr/bin/env python3
"""Convert Hugo figure shortcodes to Markdown image syntax with width 300."""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    from config_utils import get_path, get_value
except ImportError:
    from .config_utils import get_path, get_value


CONTENT_DIR = get_path("content")
DEFAULT_WIDTH = int(get_value("convert_figures.default_width", 300))


FIGURE_PATTERN = re.compile(r"{{[<%]\s*figure\s+([^{}]+?)\s*[>%]}}")
ATTR_PATTERN = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


def _normalize_src(src: str) -> str:
    src = src.strip()
    if not src:
        return src

    parsed = urlparse(src)
    if parsed.scheme in {"http", "https"}:
        return src

    path_part = parsed.path or src
    return Path(path_part).name


def replacement(match: re.Match) -> str:
    attr_text = match.group(1)
    attrs = dict(ATTR_PATTERN.findall(attr_text))

    src = attrs.get("src") or attrs.get("link") or ""
    alt = attrs.get("alt", "").strip()
    caption = attrs.get("caption", "").strip()

    alt_text = alt or caption

    alt_escaped = alt_text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
    src = _normalize_src(src)

    if not src:
        return match.group(0)

    return f"![{alt_escaped}]({src}){{ width={DEFAULT_WIDTH} }}"


def process_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    new_text, count = FIGURE_PATTERN.subn(replacement, text)
    if count == 0:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    if not CONTENT_DIR.is_dir():
        print(f"content 디렉터리를 찾을 수 없습니다: {CONTENT_DIR}", file=sys.stderr)
        return 1

    files = sorted(CONTENT_DIR.rglob("*.md"))
    total = 0
    for md_file in files:
        if process_file(md_file):
            total += 1

    print(f"변환된 파일: {total}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
