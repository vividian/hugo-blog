#!/usr/bin/env python3
"""Convert Obsidian image embeds to HTML <img> tags within Markdown files.

Usage:
  python scripts/convert_obsidian_embeds.py --content-dir path/to/content
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    from config_utils import get_path
except ImportError:
    from .config_utils import get_path

EMBED_PATTERN = re.compile(r"!\[\[([^\]]+)\]\]")
SIZE_SINGLE_PATTERN = re.compile(r"^\d+$")
SIZE_PAIR_PATTERN = re.compile(r"^(?P<width>\d+)x(?P<height>\d+)$", re.IGNORECASE)


def convert_embed(match: re.Match) -> str:
    raw = match.group(1)
    parts = [part.strip() for part in raw.split("|")]
    path = parts[0]
    filename = Path(path).name

    alt: str | None = None
    width: str | None = None
    height: str | None = None

    extras = parts[1:]

    def parse_size(value: str) -> tuple[str | None, str | None]:
        value = value.replace(" ", "")
        if SIZE_SINGLE_PATTERN.match(value):
            return value, None
        pair = SIZE_PAIR_PATTERN.match(value)
        if pair:
            return pair.group("width"), pair.group("height")
        return None, None

    if extras:
        first = extras[0]
        w, h = parse_size(first)
        if w or h:
            width, height = w, h
        else:
            alt = first
            if len(extras) > 1:
                width, height = parse_size(extras[1])
    # Set fallback alt text if nothing provided
    if not alt:
        alt = Path(filename).stem

    attrs = [f'src="{filename}"', f'alt="{alt}"']
    if width:
        attrs.append(f'width="{width}"')
    if height:
        attrs.append(f'height="{height}"')

    return f"<img {' '.join(attrs)} />"


def process_file(md_path: Path) -> bool:
    text = md_path.read_text(encoding="utf-8")
    new_text, count = EMBED_PATTERN.subn(convert_embed, text)
    if count == 0:
        return False
    md_path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--content-dir",
        type=Path,
        help="Directory containing Markdown files (default: config paths.content)",
    )
    args = parser.parse_args()

    content_dir = args.content_dir.resolve() if args.content_dir else get_path("content")
    if not content_dir.is_dir():
        print(f"지정한 디렉터리를 찾지 못했습니다: {content_dir}")
        return 1

    updated = 0
    for md_file in content_dir.rglob("*.md"):
        if process_file(md_file):
            updated += 1

    print(f"Obsidian 임베드 변환 완료: {updated}개 파일")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
