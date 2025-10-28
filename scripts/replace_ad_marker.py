#!/usr/bin/env python3
"""Replace '!광고' markers in Markdown with an AdSense shortcode."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    from config_utils import get_path, get_value
except ImportError:
    from .config_utils import get_path, get_value


MARKER = "!광고"
STANDALONE_PATTERN = re.compile(r"^!광고$", re.MULTILINE)
PARA_PATTERN = re.compile(r"<p>\s*!광고\s*</p>", re.IGNORECASE)
SHORTCODE = get_value("adsense.shortcode", "{{< adsense >}}")


def process_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARKER not in text:
        return False

    replaced = False

    new_text, count = PARA_PATTERN.subn(SHORTCODE, text)
    if count > 0:
        replaced = True
    else:
        new_text, count = STANDALONE_PATTERN.subn(SHORTCODE, text)
        if count > 0:
            replaced = True

    if not replaced:
        return False

    path.write_text(new_text, encoding="utf-8")
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

    print(f"광고 마커 변환 완료: {updated}개 파일")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
