#!/usr/bin/env python3
"""Convert config/fa.md into config/fa.yaml format."""
from __future__ import annotations

import re
from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def _resolve_path(path_str: str) -> Path:
    if not isinstance(path_str, (str, bytes)):
        return ROOT_DIR / "content/fa/fa.md"
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def _load_paths() -> tuple[Path, Path]:
    config = {}
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as fp:
            config = yaml.safe_load(fp) or {}
    paths = (config.get("financial_assets") or {}).get("paths") or {}
    md_path = _resolve_path(paths.get("fa_md", {}))
    yaml_path = _resolve_path(paths.get("fa_yaml", {}))
    return md_path, yaml_path


MD_PATH, YAML_PATH = _load_paths()


def load_markdown() -> str:
    content = MD_PATH.read_text(encoding="utf-8")
    content = content.lstrip("\ufeff")
    # remove front matter between --- ... ---
    return re.sub(r"^---[\s\S]*?---", "", content, flags=re.MULTILINE).strip()


def parse_accounts(text: str) -> list[dict]:
    accounts: list[dict] = []
    current: dict | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"계좌\s*:\s*(.+)", line)
        if m:
            name = m.group(1).strip()
            current = {"name": name, "items": []}
            accounts.append(current)
            continue
        if line.startswith("-") and current is not None:
            parts = [part.strip() for part in line[1:].split(",")]
            while len(parts) < 3:
                parts.append("")
            parts = parts[:3]
            current["items"].append(parts)
    return accounts


def _quote(value: str) -> str:
    return "\"" + value.replace("\"", "\\\"") + "\""


def write_yaml(accounts: list[dict]) -> None:
    lines: list[str] = ["accounts:"]
    for idx, account in enumerate(accounts):
        lines.append(f"  - name: {account['name']}")
        lines.append("    items:")
        if idx == 0:
            lines.append("      # [name, abbrev, ticker]")
        for name, abbrev, ticker in account["items"]:
            ticker = ticker or ""
            lines.append(
                f"      - [{_quote(name)}, {_quote(abbrev)}, {_quote(ticker)}]"
            )
    YAML_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not MD_PATH.exists():
        raise SystemExit(f"Markdown 파일을 찾을 수 없습니다: {MD_PATH}")
    text = load_markdown()
    accounts = parse_accounts(text)
    write_yaml(accounts)
    print(f"변환 완료: {MD_PATH} -> {YAML_PATH}")


if __name__ == "__main__":
    main()
