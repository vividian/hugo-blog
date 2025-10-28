#!/usr/bin/env python3
"""Utility helpers to load blog configuration."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit("PyYAML 패키지가 필요합니다. pip install pyyaml 로 설치하세요.") from exc

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


@functools.lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_path(key: str) -> Path:
    config = load_config()
    paths = config.get("paths", {})
    value = paths.get(key)
    if value is None:
        raise KeyError(f"paths.{key} is not configured")
    return (ROOT / value).resolve()


def get_value(path: str, default: Any = None) -> Any:
    config = load_config()
    node: Any = config
    for part in path.split('.'):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return node
