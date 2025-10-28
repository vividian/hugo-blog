#!/usr/bin/env python3
"""Normalize Hugo front matter for Markdown files in blog/content.

For each Markdown file, we:
  - Keep only selected fields (title, date, author, slug, url, aliases,
    description, tags, categories, keywords, summary) in that order.
  - Ensure string fields are quoted.
  - Convert tags/categories to JSON-style arrays; keywords mirrors tags.
  - Generate aliases from existing url/guid values.
  - Leave the body content untouched.

PyYAML is required (pip install pyyaml).
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    print("PyYAML이 필요합니다. pip install pyyaml 후 다시 실행하세요.", file=sys.stderr)
    sys.exit(1)


ROOT = Path(__file__).resolve().parents[1]  # blog/
CONTENT_DIR = ROOT / "content"


def read_front_matter(lines: list[str]) -> tuple[dict, int]:
    """Return (front_matter_dict, closing_index)."""
    if not lines or lines[0].strip() != "---":
        return {}, -1

    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            closing_index = idx
            break
    else:
        return {}, -1

    fm_text = "\n".join(lines[1:closing_index])
    if not fm_text.strip():
        return {}, closing_index

    try:
        data = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML 파싱 실패: {exc}") from exc

    if not isinstance(data, dict):
        data = {}

    return data, closing_index


def to_string(value) -> str:
    if value is None:
        return ""
    return str(value)


def normalize_sequence(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        # If the string already looks like a YAML/JSON array, try loading again.
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = yaml.safe_load(stripped)
                return normalize_sequence(parsed)
            except yaml.YAMLError:
                pass
        pieces = [part.strip() for part in re.split(r"[,\n]+", value)]
        return [p for p in pieces if p]
    if isinstance(value, Iterable):
        result = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                result.append(text)
        return result
    return [str(value).strip()]


def make_slug(title: str, fallback: str) -> str:
    base = title.strip() or fallback
    base = re.sub(r"\s+", "-", base)
    base = re.sub(r"-+", "-", base)
    return base.strip("-")


def quote_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def format_array(items: list[str]) -> str:
    return json.dumps(items, ensure_ascii=False)


def build_front_matter(
    data: dict,
    md_path: Path,
) -> list[str]:
    title = to_string(data.get("title")).strip()
    date = to_string(data.get("date")).strip()
    author = to_string(data.get("author")).strip()
    url = to_string(data.get("url")).strip()

    tags = normalize_sequence(data.get("tags") or data.get("tag"))
    categories = normalize_sequence(data.get("categories") or data.get("category"))

    # Build aliases preserving order, deduplicated.
    alias_candidates = []
    if url:
        alias_candidates.append(url)

    existing_aliases = normalize_sequence(data.get("aliases") or data.get("alias"))
    alias_candidates.extend(existing_aliases)

    guid = to_string(data.get("guid")).strip()
    if guid:
        alias_candidates.append(guid)

    fallback_slug = md_path.stem
    if md_path.name == "index.md":
        fallback_slug = md_path.parent.name or fallback_slug
    slug = make_slug(title, fallback_slug)

    aliases_with_slash: list[str] = []
    for candidate in alias_candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        if candidate.startswith("http://") or candidate.startswith("https://"):
            continue
        if not candidate.endswith("/"):
            candidate = f"{candidate}/"
        aliases_with_slash.append(candidate)

    slug_alias = f"/{slug.strip('/')}/" if slug else ""
    if slug_alias and slug_alias not in aliases_with_slash:
        aliases_with_slash.append(slug_alias)

    aliases_with_slash = list(dict.fromkeys(aliases_with_slash))

    front_lines = [
        "---",
        f"title: {quote_string(title)}",
        f"date: {quote_string(date)}",
        f"author: {quote_string(author)}",
        f"slug: {quote_string(slug)}",
        f"url: {quote_string(url)}",
    ]

    if aliases_with_slash:
        front_lines.append("aliases:")
        for alias in aliases_with_slash:
            front_lines.append(f'  - {quote_string(alias)}')
    else:
        front_lines.append("aliases: []")

    front_lines.extend(
        [
        'description: ""',
        f"tags: {format_array(tags)}",
        f"categories: {format_array(categories)}",
        f"keywords: {format_array(tags)}",
        'summary: ""',
        "---",
    ]

    return front_lines


def process_markdown(md_path: Path) -> bool:
    raw = md_path.read_text(encoding="utf-8")
    if raw.startswith("\ufeff"):
        raw = raw.lstrip("\ufeff")

    lines = raw.split("\n")
    try:
        front_data, closing_idx = read_front_matter(lines)
    except ValueError as exc:
        print(f"[경고] {md_path}: {exc}", file=sys.stderr)
        return False

    if closing_idx == -1:
        print(f"[건너뜀] 프론트매터가 없거나 잘못된 파일: {md_path}", file=sys.stderr)
        return False

    body_lines = lines[closing_idx + 1 :]

    new_front_lines = build_front_matter(front_data, md_path)
    new_lines = new_front_lines + body_lines
    new_content = "\n".join(new_lines)
    if not new_content.endswith("\n"):
        new_content += "\n"

    md_path.write_text(new_content, encoding="utf-8")
    return True


def main() -> int:
    md_files = sorted(CONTENT_DIR.rglob("*.md"))
    if not md_files:
        print("Markdown 파일을 찾지 못했습니다.", file=sys.stderr)
        return 1

    success = 0
    for md_file in md_files:
        if process_markdown(md_file):
            success += 1

    print(f"완료: {success}개 파일 업데이트")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
