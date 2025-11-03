#!/usr/bin/env python3
"""Convert <img> tags in Markdown files to Obsidian-style embeds."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import urlparse

try:
    from config_utils import get_path
except ImportError:
    from .config_utils import get_path

CONTENT_DIR = get_path("content")
IMG_TAG_PATTERN = re.compile(r"<img\s+[^>]*?>", re.IGNORECASE)
ATTR_PATTERN = re.compile(r"(\w+)\s*=\s*\"([^\"]*)\"")
STYLE_PROP_PATTERN = re.compile(r"([a-zA-Z-]+)\s*:\s*([^;]+)")


def is_remote_src(src: str) -> bool:
    parsed = urlparse(src)
    return parsed.scheme in {"http", "https"}


def cleanup_dimension(value: str) -> str:
    value = value.strip()
    if value.endswith("px"):
        value = value[:-2]
    return value


def extract_dimensions(attrs: dict[str, str]) -> tuple[str | None, str | None]:
    width = attrs.get("width")
    height = attrs.get("height")

    style = attrs.get("style", "")
    if style:
        for prop, val in STYLE_PROP_PATTERN.findall(style):
            prop = prop.lower()
            if prop == "width" and not width:
                width = val
            elif prop == "height" and not height:
                height = val

    if width:
        width = cleanup_dimension(width)
    if height:
        height = cleanup_dimension(height)
    return width, height


def convert_img_tag(tag: str, md_dir: Path) -> str:
    attrs = dict((name.lower(), value) for name, value in ATTR_PATTERN.findall(tag))
    src = attrs.get("src")
    if not src:
        return tag

    display_src = src
    if not is_remote_src(src):
        display_src = Path(src).name

    alt = attrs.get("alt", "").strip()

    width, height = extract_dimensions(attrs)
    size_part = None
    if width and height:
        size_part = f"{width}x{height}"
    elif width:
        size_part = width
    elif height:
        size_part = height

    parts = [display_src]
    include_alt = bool(alt)
    if size_part and not include_alt:
        # maintain placeholder for empty alt so size occupies third slot
        include_alt = True
        alt = ""
    if include_alt:
        parts.append(alt)
    if size_part:
        parts.append(size_part)

    return f"![[{'|'.join(parts)}]]"


def process_markdown_file(md_path: Path) -> bool:
    text = md_path.read_text(encoding="utf-8")
    changed = False

    def repl(match: re.Match[str]) -> str:
        nonlocal changed
        tag = match.group(0)
        replacement = convert_img_tag(tag, md_path.parent)
        if replacement != tag:
            changed = True
            return replacement
        return tag

    new_text = IMG_TAG_PATTERN.sub(repl, text)
    if changed:
        md_path.write_text(new_text, encoding="utf-8")
    return changed


def main() -> int:
    if not CONTENT_DIR.is_dir():
        print("content 디렉터리를 찾을 수 없습니다", file=sys.stderr)
        return 1

    updated = 0
    for md_file in CONTENT_DIR.rglob("*.md"):
        if process_markdown_file(md_file):
            updated += 1

    print(f"Converted <img> tags in {updated} Markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
