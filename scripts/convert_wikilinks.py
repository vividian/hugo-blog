#!/usr/bin/env python3
"""Convert Obsidian-style wiki links ([[...]] ) to HTML anchors in public pages."""

from __future__ import annotations

import html
import re
from pathlib import Path

try:
    from config_utils import get_path, get_value
except ImportError:
    from .config_utils import get_path, get_value

CONTENT_DIR = get_path("content")
PUBLIC_DIR = get_path("public")
PERMALINKS = get_value("wikilinks.permalinks", {}) or {}

WIKILINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[\s]+", "-", value)
    value = re.sub(r"[^0-9a-z\-가-힣_]+", "", value)
    return value


def parse_front_matter(md_path: Path) -> dict[str, str]:
    text = md_path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    front = parts[1]
    result: dict[str, str] = {}
    for key in ("title", "slug", "url"):
        pattern = re.compile(rf"^{key}:\s*[\"']?([^\"'\n]+)", re.MULTILINE)
        match = pattern.search(front)
        if match:
            result[key] = match.group(1).strip()
    return result


def build_mapping() -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for md_path in CONTENT_DIR.rglob("index.md"):
        rel_parts = md_path.relative_to(CONTENT_DIR).parts
        if len(rel_parts) < 2:
            continue
        section = rel_parts[0]
        folder_name = rel_parts[-2]
        fm = parse_front_matter(md_path)
        slug = fm.get("slug") or slugify(folder_name)
        url = fm.get("url")
        if not url:
            pattern = PERMALINKS.get(section)
            if pattern:
                url = pattern.replace(":slug", slug)
            else:
                url = f"/{section}/{slug}/"
        keys = {
            folder_name.strip(): url,
            folder_name.strip().lower(): url,
            slug: url,
        }
        title = fm.get("title")
        if title:
            keys[title.strip()] = url
            keys[title.strip().lower()] = url
        entry = {"url": url, "title": fm.get("title") or folder_name}
        for key in keys:
            if key and key not in mapping:
                mapping[key] = entry
    return mapping


def find_entry(mapping: dict[str, dict[str, str]], target: str) -> dict[str, str] | None:
    if target in mapping:
        return mapping[target]
    lowered = target.lower()
    if lowered in mapping:
        return mapping[lowered]
    slugged = slugify(target)
    if slugged in mapping:
        return mapping[slugged]
    return None


def convert_file(path: Path, mapping: dict[str, str]) -> bool:
    text = path.read_text(encoding="utf-8")
    body_match = re.search(r"(<body[^>]*>)(.*)(</body>)", text, flags=re.IGNORECASE | re.DOTALL)
    if not body_match:
        return False

    body_start = body_match.start(2)
    body_end = body_match.end(2)
    changed = False

    def repl(match: re.Match[str]) -> str:
        nonlocal changed
        raw = match.group(1)
        parts = [p.strip() for p in raw.split("|", 1)]
        target = parts[0]
        custom_label = parts[1] if len(parts) > 1 else None
        entry = find_entry(mapping, target)
        label = custom_label or (entry.get("title") if entry else target)

        if body_start <= match.start() < body_end:
            if entry:
                changed = True
                url = entry["url"]
                return f'<a href="{html.escape(url)}">{html.escape(label)}</a>'
            else:
                changed = True
                return html.escape(label)
        else:
            return html.escape(label)

    new_text = WIKILINK_PATTERN.sub(repl, text)
    if changed:
        path.write_text(new_text, encoding="utf-8")
    return changed


def main() -> int:
    if not PUBLIC_DIR.is_dir():
        print("public 디렉터리를 찾을 수 없습니다")
        return 1

    mapping = build_mapping()
    if not mapping:
        print("위키 링크 매핑 정보가 없습니다")

    converted = 0
    for html_file in PUBLIC_DIR.rglob("*.html"):
        if convert_file(html_file, mapping):
            converted += 1

    print(f"위키 링크 변환 완료: {converted}개 파일")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
