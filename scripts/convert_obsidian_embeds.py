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

EMBED_PATTERN = re.compile(r"!\[\[([^|\]]+)(?:\|([^\]]+))?\]\]")


def convert_embed(match: re.Match) -> str:
    path = match.group(1).strip()
    size = (match.group(2) or "").strip()
    alt = Path(path).name

    styles: list[str] = []
    if size:
        if size.isdigit():
            styles.append(f"width:{size}px")
        elif re.match(r"^\d+x\d+$", size):
            width, height = size.split("x", 1)
            if width.isdigit():
                styles.append(f"width:{width}px")
            if height.isdigit():
                styles.append(f"height:{height}px")
        else:
            styles.append(f"width:{size}")

    style_attr = f' style="{"; ".join(styles)}"' if styles else ""
    return f'<img src="{path}" alt="{alt}"{style_attr} />'


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
