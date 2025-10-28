#!/usr/bin/env python3
"""Remove literal percent signs from slug, url, and aliases entries."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTENT_DIR = ROOT / "content"

ALIAS_ITEM_RE = re.compile(r'^(\s*-\s*")([^"]*)(".*)$')


REMOVALS = {"%", "\\", '"', "'", "“", "”", "‘", "’"}


def clean_value(value: str) -> str:
    for ch in REMOVALS:
        value = value.replace(ch, "")
    return value


def normalize_component(value: str) -> str:
    value = clean_value(value)
    value = value.strip()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"-+", "-", value)
    value = re.sub(r"[^0-9A-Za-z가-힣\-]", "-", value)
    return value.strip("-")


def normalize_path(value: str, fallback: str) -> str:
    value = clean_value(value).strip()
    value = value.strip("/")
    if not value:
        value = fallback
    segments = [normalize_component(seg) for seg in value.split("/") if seg]
    path = "/".join(seg for seg in segments if seg)
    if not path:
        path = fallback
    return f"/{path}/"


def process_file(path: Path) -> bool:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return False

    changed = False
    inside_front = True
    inside_aliases = False
    current_slug = ""

    for idx, line in enumerate(lines):
        if inside_front and line.strip() == "---" and idx != 0:
            inside_front = False
            inside_aliases = False
            continue

        if not inside_front:
            break

        stripped = line.lstrip()
        if stripped.startswith("slug:"):
            raw = line.split(":", 1)[1].strip()
            if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
                raw = raw[1:-1]
            elif raw.startswith("'") and raw.endswith("'") and len(raw) >= 2:
                raw = raw[1:-1]
            cleaned = normalize_component(raw)
            if cleaned != raw:
                changed = True
            current_slug = cleaned or current_slug
            lines[idx] = f'slug: "{current_slug}"'
            continue

        if stripped.startswith("url:"):
            raw = line.split(":", 1)[1].strip()
            if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
                raw_value = raw[1:-1]
            elif raw.startswith("'") and raw.endswith("'") and len(raw) >= 2:
                raw_value = raw[1:-1]
            else:
                raw_value = raw
            normalized = normalize_path(raw_value, current_slug)
            if normalized != raw_value:
                changed = True
            lines[idx] = f'url: "{normalized}"'
            continue

        if line.startswith("aliases:"):
            inside_aliases = True
            continue

        if inside_aliases:
            if not line.startswith(" ") and not line.startswith("\t"):
                inside_aliases = False
            else:
                match = ALIAS_ITEM_RE.match(line)
                if match:
                    raw = match.group(2)
                    normalized = normalize_path(raw, current_slug)
                    if normalized != raw:
                        changed = True
                    lines[idx] = f'{match.group(1)}{normalized}{match.group(3)}'

    if changed:
        lines.append("")  # ensure trailing newline
        path.write_text("\n".join(lines), encoding="utf-8")
    return changed


def main() -> int:
    updated = 0
    for md_path in CONTENT_DIR.rglob("*.md"):
        if process_file(md_path):
            updated += 1
    print(f"Percent 제거 파일 수: {updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
