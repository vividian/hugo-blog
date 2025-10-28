#!/usr/bin/env bash
# Hugo 프론트매터를 정리하는 쉘 스크립트.
# blog/content 아래의 모든 Markdown 파일을 순회하며 워드프레스에서 넘어온
# 프론트매터를 지정한 형식으로 재구성한다.
# PyYAML(pip install pyyaml)이 설치되어 있어야 한다.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONTENT_DIR="${ROOT_DIR}/content"

if [[ ! -d "${CONTENT_DIR}" ]]; then
  echo "content 디렉터리를 찾을 수 없습니다: ${CONTENT_DIR}" >&2
  exit 1
fi

python3 - <<'PYTHON' "${CONTENT_DIR}"
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
    raise SystemExit(1)


CONTENT_DIR = Path(sys.argv[1])


def read_front_matter(lines: list[str]) -> tuple[dict, int]:
    if not lines or lines[0].strip() != "---":
        return {}, -1
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            closing = idx
            break
    else:
        return {}, -1

    block = "\n".join(lines[1:closing])
    if not block.strip():
        return {}, closing

    try:
        data = yaml.safe_load(block) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML 파싱 실패: {exc}") from exc
    if not isinstance(data, dict):
        data = {}
    return data, closing


def to_str(value) -> str:
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
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = yaml.safe_load(stripped)
                return normalize_sequence(parsed)
            except yaml.YAMLError:
                pass
        items = [part.strip() for part in re.split(r"[,\n]+", value)]
        return [itm for itm in items if itm]
    if isinstance(value, Iterable):
        cleaned: list[str] = []
        for itm in value:
            if itm is None:
                continue
            text = str(itm).strip()
            if text:
                cleaned.append(text)
        return cleaned
    return [str(value).strip()]


def make_slug(title: str, fallback: str) -> str:
    base = title.strip() or fallback
    base = re.sub(r"\s+", "-", base)
    base = re.sub(r"-+", "-", base)
    return base.strip("-")


def quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def format_array(items: list[str]) -> str:
    return json.dumps(items, ensure_ascii=False)


def rebuild_front_matter(data: dict, md_path: Path) -> list[str]:
    title = to_str(data.get("title")).strip()
    date = to_str(data.get("date")).strip()
    author = to_str(data.get("author")).strip()
    url = to_str(data.get("url")).strip()
    guid = to_str(data.get("guid")).strip()

    tags = normalize_sequence(data.get("tags") or data.get("tag"))
    categories = normalize_sequence(data.get("categories") or data.get("category"))

    aliases: list[str] = []
    if url:
        aliases.append(url)
    aliases.extend(normalize_sequence(data.get("aliases") or data.get("alias")))
    if guid:
        aliases.append(guid)

    fallback_slug = md_path.stem
    if md_path.name == "index.md":
        fallback_slug = md_path.parent.name or fallback_slug
    slug = make_slug(title, fallback_slug)

    aliases_with_slash: list[str] = []
    for candidate in aliases:
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

    front = [
        "---",
        f"title: {quote(title)}",
        f"date: {quote(date)}",
        f"author: {quote(author)}",
        f"slug: {quote(slug)}",
        f"url: {quote(url)}",
    ]

    if aliases_with_slash:
        front.append("aliases:")
        front.extend(f'  - {quote(alias)}' for alias in aliases_with_slash)
    else:
        front.append("aliases: []")

    front.extend(
        [
            'description: ""',
            f"tags: {format_array(tags)}",
            f"categories: {format_array(categories)}",
            f"keywords: {format_array(tags)}",
            'summary: ""',
            "---",
        ]
    )
    return front


def process_file(md_path: Path) -> bool:
    raw = md_path.read_text(encoding="utf-8")
    if raw.startswith("\ufeff"):
        raw = raw.lstrip("\ufeff")
    lines = raw.split("\n")
    try:
        data, closing = read_front_matter(lines)
    except ValueError as exc:
        print(f"[경고] {md_path}: {exc}", file=sys.stderr)
        return False
    if closing == -1:
        print(f"[건너뜀] 프론트매터 없음: {md_path}", file=sys.stderr)
        return False
    body = lines[closing + 1 :]
    new_front = rebuild_front_matter(data, md_path)
    new_lines = new_front + body
    new_content = "\n".join(new_lines)
    if not new_content.endswith("\n"):
        new_content += "\n"
    md_path.write_text(new_content, encoding="utf-8")
    return True


md_files = sorted(CONTENT_DIR.rglob("*.md"))
if not md_files:
    print("Markdown 파일을 찾지 못했습니다.", file=sys.stderr)
    raise SystemExit(1)

count = 0
for md_path in md_files:
    if process_file(md_path):
        count += 1

print(f"완료: {count}개 파일 업데이트")
PYTHON
