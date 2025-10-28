#!/usr/bin/env python3
"""Replace aliases front matter with the URL value for all Markdown files."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTENT_DIR = ROOT / "content"


def ensure_trailing_slash(value: str) -> str:
    if not value:
        return value
    return value if value.endswith("/") else value + "/"


def main() -> int:
    files = sorted(CONTENT_DIR.rglob("*.md"))
    updated = 0

    for path in files:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            continue

        end = None
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                end = idx
                break

        if end is None:
            continue

        fm_lines = lines[1:end]
        body_lines = lines[end + 1 :]

        fm_text = "\n".join(fm_lines)
        url_match = re.search(r'^url:\s*"([^"]*)"', fm_text, re.MULTILINE)
        url_value = url_match.group(1) if url_match else ""

        slug_match = re.search(r'^slug:\s*"([^"]*)"', fm_text, re.MULTILINE)
        slug_value = slug_match.group(1) if slug_match else ""

        alias_values = []
        if url_value:
            url_with_slash = ensure_trailing_slash(url_value.strip())
            if url_with_slash:
                alias_values.append(url_with_slash)

        if slug_value:
            slug_clean = slug_value.strip("/")
            if slug_clean:
                slug_alias = ensure_trailing_slash(f"/{slug_clean}")
                if slug_alias not in alias_values:
                    alias_values.append(slug_alias)

        alias_values = list(dict.fromkeys(alias_values))

        cleaned_fm_lines = []
        skipping_alias = False
        for line in fm_lines:
            if skipping_alias:
                if (
                    line.startswith(" ")
                    or line.startswith("\t")
                    or line.strip().startswith("-")
                    or not line.strip()
                ):
                    continue
                skipping_alias = False
            if skipping_alias:
                continue
            if re.match(r"aliases\s*:", line):
                skipping_alias = True
                continue
            cleaned_fm_lines.append(line)

        insert_index = len(cleaned_fm_lines)
        for idx, line in enumerate(cleaned_fm_lines):
            if line.startswith("url:"):
                insert_index = idx + 1
                break

        if alias_values:
            alias_lines = ["aliases:"] + [f'  - "{val if val.endswith("/") else val + "/"}"' for val in alias_values]
        else:
            alias_lines = ["aliases: []"]

        new_fm_lines = (
            cleaned_fm_lines[:insert_index] + alias_lines + cleaned_fm_lines[insert_index:]
        )

        new_lines = ["---"] + new_fm_lines + ["---"] + body_lines

        new_text = "\n".join(new_lines)
        if text != new_text:
            path.write_text(new_text, encoding="utf-8")
            updated += 1

    print(f"aliases 업데이트 완료: {updated}개 파일")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
