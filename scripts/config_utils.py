#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""블로그 설정(configuration)을 불러오기 위한 유틸리티 헬퍼 함수들."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

# 이 파일의 상위 디렉터리(프로젝트 루트)를 ROOT 경로로 설정합니다.
ROOT = Path(__file__).resolve().parents[1]
# 기본 설정 파일의 경로를 지정합니다.
BASE_CONFIG_PATH = ROOT / "hugo.yaml"
# 민감한 정보나 추가 설정을 담고 있는 비밀 설정 파일의 후보 경로들을 리스트로 정의합니다.
SECRET_CONFIG_CANDIDATES = [
    ROOT / "config" / "config.yaml",
]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    두 개의 딕셔너리를 재귀적으로 병합합니다.
    - `override` 딕셔너리의 값이 `base` 딕셔너리의 값을 덮어씁니다.
    - 만약 두 딕셔너리 모두에 같은 키가 있고 그 값이 또 다른 딕셔너리이면,
      그 하위 딕셔너리에 대해서도 재귀적으로 병합을 수행합니다.
    """
    merged = dict(base)  # 기본 딕셔너리의 복사본으로 시작합니다.
    for key, override_value in override.items():
        base_value = merged.get(key)
        # 키에 해당하는 값이 두 딕셔너리 모두에서 딕셔너리인 경우, 재귀적으로 병합합니다.
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = _deep_merge(base_value, override_value)
        # 그렇지 않은 경우, override 값으로 덮어씁니다.
        else:
            merged[key] = override_value
    return merged


@functools.lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """
    기본 설정 파일과 비밀 설정 파일을 모두 불러와 병합한 최종 설정을 반환합니다.
    - `lru_cache`를 사용하여 함수의 결과를 캐싱합니다. 즉, 한 번 실행된 후에는 다시 파일을 읽지 않고
      메모리에 저장된 결과를 즉시 반환하여 성능을 향상시킵니다.
    """
    # 1. 기본 설정 파일(hugo.yaml)을 읽습니다.
    with BASE_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        base_config = yaml.safe_load(fh) or {}

    # 2. 비밀 설정 파일 후보들을 순회하며, 존재하는 파일을 찾아 기본 설정에 덮어씁니다.
    for cfg_path in SECRET_CONFIG_CANDIDATES:
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as fh:
                secret_config = yaml.safe_load(fh) or {}
            # 깊은 병합을 통해 설정을 합칩니다 (비밀 설정이 우선순위를 가집니다).
            base_config = _deep_merge(base_config, secret_config)

    return base_config


def get_path(key: str) -> Path:
    """
    설정에서 'paths' 항목 아래의 특정 `key`에 해당하는 경로를 가져와 절대 경로(Path 객체)로 반환합니다.
    """
    config = load_config()
    paths = config.get("paths", {})
    value = paths.get(key)
    if value is None:
        raise KeyError(f"paths.{key} is not configured")
    return (ROOT / value).resolve()


def get_value(path: str, default: Any = None) -> Any:
    """
    설정에서 점(.)으로 구분된 경로(`path`)를 이용해 특정 값을 가져옵니다.
    - 예: `get_value("ssh_deploy.host")`는 `config['ssh_deploy']['host']` 값을 찾습니다.
    - 경로가 존재하지 않으면 `default` 값을 반환합니다.
    """
    config = load_config()
    node: Any = config
    for part in path.split('.'):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            # 경로의 일부라도 존재하지 않으면 즉시 default 값을 반환합니다.
            return default
    return node
