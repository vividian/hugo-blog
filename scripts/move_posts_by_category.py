#!/usr/bin/env python3
"""Move post bundles from content/posts to category-specific sections."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    print("PyYAML이 필요합니다. 먼저 `pip install pyyaml`을 실행하세요.", file=sys.stderr)
    raise SystemExit(1)


ROOT = Path(__file__).resolve().parents[1]
CONTENT_DIR = ROOT / "content"
POSTS_DIR = CONTENT_DIR / "posts"

CATEGORY_MAP = [
    ("사용기", "reviews"),
    ("시놀로지", "synology"),
    ("식도락", "gourmet"),
    ("여행", "travel"),
    ("일상", "daily"),
    ("적바림", "notes"),
    ("주식", "stocks"),
    ("배당기록", "stocks/dividends"),
]


def read_categories(index_path: Path) -> list[str]:
    text = index_path.read_text(encoding="utf-8")
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")

    parts = text.split("---")
    if len(parts) < 3:
        return []

    front_matter = parts[1]
    try:
        data = yaml.safe_load(front_matter) or {}
    except yaml.YAMLError:
        return []

    categories = data.get("categories") or []
    if isinstance(categories, str):
        categories = [categories]

    cleaned = []
    for item in categories:
        if item is None:
            continue
        cleaned.append(str(item).strip())
    return cleaned


def find_target(categories: list[str]) -> str | None:
    for cat in categories:
        for key, dest in CATEGORY_MAP:
            if cat == key:
                return dest
    return None


def main() -> int:
    if not POSTS_DIR.is_dir():
        print(f"디렉터리를 찾지 못했습니다: {POSTS_DIR}", file=sys.stderr)
        return 1

    bundles = [p for p in POSTS_DIR.iterdir() if p.is_dir()]
    if not bundles:
        print("posts 디렉터리에 번들 폴더가 없습니다.")
        return 0

    moved = 0
    skipped = 0

    for bundle in bundles:
        index_md = bundle / "index.md"
        if not index_md.is_file():
            print(f"index.md가 없어 건너뜀: {bundle}", file=sys.stderr)
            skipped += 1
            continue

        categories = read_categories(index_md)
        target = find_target(categories)
        if not target:
            skipped += 1
            continue

        target_dir = CONTENT_DIR.joinpath(*target.split("/"))
        dest = target_dir / bundle.name

        if dest.exists():
            print(f"이미 대상 폴더가 존재합니다. 건너뜀: {dest}", file=sys.stderr)
            skipped += 1
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(bundle), dest)
        print(f"{bundle.relative_to(ROOT)} -> {dest.relative_to(ROOT)}")
        moved += 1

    print(f"이동 완료: {moved}개, 건너뜀: {skipped}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
