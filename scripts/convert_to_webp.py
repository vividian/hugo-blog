#!/usr/bin/env python3
"""Convert images under content/ to WebP and update Markdown references."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Dict

try:
    from PIL import Image
except ModuleNotFoundError:
    print("Pillow 라이브러리가 필요합니다. python -m pip install Pillow 로 설치하세요.", file=sys.stderr)
    sys.exit(1)

try:
    from config_utils import get_path
except ImportError:
    from .config_utils import get_path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff"}
POSSIBLE_ORIGINAL_EXTS = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff"]
TARGET_WIDTH = 720
WEBP_QUALITY = 85

OBSIDIAN_EMBED_PATTERN = re.compile(r"(!?)\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")

PREFIXES_TO_SKIP = ("http://", "https://", "//", "data:", "mailto:")


def normalize_filename(path: Path) -> Path:
    new_name = path.name.replace(" ", "_")
    if new_name != path.name:
        new_path = path.with_name(new_name)
        path.rename(new_path)
        return new_path
    return path


def register_mapping(mapping: Dict[str, str], original: PurePosixPath, dest: PurePosixPath) -> None:
    def _set(key: str, value: str) -> None:
        if key and key not in mapping:
            mapping[key] = value

    original_posix = original.as_posix()
    dest_posix = dest.as_posix()

    base_keys = [
        original_posix,
        original.name,
        original_posix.replace("_", " "),
        original.name.replace("_", " "),
    ]
    for key in base_keys:
        _set(key, dest_posix if "/" in key else dest.name)

    prefix_aliases = [
        f"content/{original_posix}",
        f"./content/{original_posix}",
        f"blog/content/{original_posix}",
        f"./blog/content/{original_posix}",
    ]
    for alias in prefix_aliases:
        _set(alias, dest_posix)


def convert_image(path: Path, base: Path, mapping: Dict[str, str]) -> bool:
    if path.suffix.lower() not in VALID_EXTENSIONS:
        return False

    original_rel = PurePosixPath(path.relative_to(base).as_posix())
    normalized_path = normalize_filename(path)
    normalized_rel = PurePosixPath(normalized_path.relative_to(base).as_posix())

    dest = normalized_path.with_suffix(".webp")
    dest_rel = PurePosixPath(dest.relative_to(base).as_posix())

    with Image.open(normalized_path) as img:
        width, height = img.size
        if width > TARGET_WIDTH:
            new_height = int(height * TARGET_WIDTH / width)
            img = img.resize((TARGET_WIDTH, new_height), Image.LANCZOS)
        img.save(dest, "WEBP", quality=WEBP_QUALITY)

    normalized_path.unlink()

    register_mapping(mapping, original_rel, dest_rel)
    register_mapping(mapping, normalized_rel, dest_rel)
    return True


def register_existing_webp(path: Path, base: Path, mapping: Dict[str, str]) -> None:
    dest_rel = PurePosixPath(path.relative_to(base).as_posix())
    stem = dest_rel.with_suffix("")
    for ext in POSSIBLE_ORIGINAL_EXTS:
        original = stem.with_suffix(ext)
        register_mapping(mapping, original, dest_rel)


def resolve_relative(base: PurePosixPath, target: PurePosixPath) -> PurePosixPath:
    if target.is_absolute():
        return target.relative_to("/")
    parts = list(base.parts)
    for part in target.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
        else:
            parts.append(part)
    return PurePosixPath(*parts)


def update_markdown(md_file: Path, base: Path, mapping: Dict[str, str]) -> bool:
    text = md_file.read_text(encoding="utf-8")
    changed = False
    md_dir_rel = PurePosixPath(md_file.parent.relative_to(base).as_posix())

    def embed_repl(match: re.Match[str]) -> str:
        nonlocal changed
        bang, target = match.groups()
        if target.startswith(PREFIXES_TO_SKIP):
            return match.group(0)

        candidates = [target, target.replace(" ", "_")]
        for candidate in candidates:
            candidate_path = PurePosixPath(candidate)
            full_rel = resolve_relative(md_dir_rel, candidate_path)
            key_options = [full_rel.as_posix(), full_rel.name, candidate, candidate.replace("_", " ")]
            for key in key_options:
                if key in mapping:
                    dest = PurePosixPath(mapping[key])
                    new_url = dest.name
                    changed = True
                    return match.group(0).replace(target, new_url, 1)

        if ("/" in target or "\\" in target) and not target.startswith(PREFIXES_TO_SKIP):
            suffix = PurePosixPath(target).suffix.lower()
            if suffix in VALID_EXTENSIONS or suffix == ".webp":
                basename = PurePosixPath(target).name
                if basename and basename != target:
                    changed = True
                    return match.group(0).replace(target, basename, 1)
        return match.group(0)

    text = OBSIDIAN_EMBED_PATTERN.sub(embed_repl, text)

    if changed:
        md_file.write_text(text, encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content-dir", type=Path, help="Content directory (default: config paths.content)")
    args = parser.parse_args()

    content_dir = args.content_dir.resolve() if args.content_dir else get_path("content")
    if not content_dir.is_dir():
        print(f"{content_dir} 디렉터리를 찾을 수 없습니다", file=sys.stderr)
        return 1

    mapping: Dict[str, str] = {}
    converted = 0
    for path in list(content_dir.rglob("*")):
        if path.is_file():
            if convert_image(path, content_dir, mapping):
                converted += 1

    # Register existing WebP files to cover cases where images were already converted.
    for webp in content_dir.rglob("*.webp"):
        register_existing_webp(webp, content_dir, mapping)

    updated = 0
    for md_file in content_dir.rglob("index.md"):
        if update_markdown(md_file, content_dir, mapping):
            updated += 1

    print(f"Converted {converted} images to WebP. Updated {updated} Markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
