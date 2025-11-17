#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NAS 환경에서 Hugo 사이트(포트폴리오 포함)를 빌드·배포하는 스크립트."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
import shutil

from scripts import fetch_top_posts
from scripts.config_utils import get_path, get_value, load_config

# --- 전역 변수 설정 ---
# 스크립트 파일의 절대 경로를 기준으로 루트 디렉터리를 정의합니다.
ROOT = Path(__file__).resolve().parent
# 현재 시스템에서 사용 중인 Python 인터프리터의 경로를 저장합니다.
PYTHON = sys.executable


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    """
    주어진 셸 명령어를 실행하고, 실행 과정을 표준 출력으로 보여줍니다.
    - `cwd` 인자를 통해 작업 디렉터리를 지정할 수 있습니다. (기본값: ROOT)
    - 명령어 실행 실패 시 `subprocess.CalledProcessError` 예외를 발생시킵니다.
    """
    print("$", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=cwd or ROOT)


def run_python(script: str, *args: str) -> None:
    """
    프로젝트 루트 디렉터리에서 지정된 파이썬 스크립트를 실행합니다.
    - `run` 함수를 내부적으로 사용하여 일관된 실행 방식을 유지합니다.
    """
    run([PYTHON, script, *args], cwd=ROOT)


def clean_metadata_dirs(paths: list[Path]) -> None:
    """
    Synology NAS 환경에서 자동으로 생성되는 `@eaDir` 같은 메타데이터 폴더를 재귀적으로 찾아 제거합니다.
    - 지정된 경로 리스트(`paths`) 내에 존재하는 모든 하위 `@eaDir`를 삭제합니다.
    - 폴더 삭제 중 오류 발생 시, 경고 메시지를 출력하고 계속 진행합니다.
    """
    for base in paths:
        if not base.exists():
            continue
        for ea_dir in base.rglob("@eaDir"):
            if ea_dir.is_dir():
                try:
                    shutil.rmtree(ea_dir)
                    print(f"삭제: {ea_dir}")
                except Exception as exc:
                    print(f"(경고) {ea_dir} 제거 실패: {exc}")


def parse_args() -> argparse.Namespace:
    """
    스크립트 실행 시 필요한 명령줄 인자를 파싱하여 반환합니다.
    - `--full-portfolio`: 전체 기간의 포트폴리오를 재계산할지 여부.
    - `--target`: 빌드 결과물(public)을 복사할 로컬 경로.
    """
    parser = argparse.ArgumentParser(description="NAS에서 Hugo 사이트 배포")
    parser.add_argument(
        "--full-portfolio",
        action="store_true",
        help="포트폴리오 전체 기간을 재계산합니다 (기본: 최신 월만 계산)",
    )
    parser.add_argument(
        "--target",
        type=Path,
        help="빌드된 public을 복사할 로컬 경로 (지정하지 않으면 동기화를 생략합니다)",
    )
    return parser.parse_args()


def build_site(full_portfolio: bool) -> Path:
    """
    사이트 빌드 전체 과정을 수행하고, 빌드된 `public` 디렉터리의 경로를 반환합니다.
    1. 포트폴리오 데이터 업데이트 (`update_fa.py`)
    2. 인기 포스트 데이터 업데이트 (`fetch_top_posts.py`)
    3. 임시 디렉터리 생성 및 원본 콘텐츠 복사
    4. 콘텐츠 전처리 (이미지 변환, 위키링크 변환 등)
    5. Hugo 빌드 실행
    6. 빌드 후처리 (public 디렉터리 내 위키링크 변환)
    """
    # 1. 포트폴리오 데이터 업데이트
    try:
        run_python("scripts/convert_fa_md.py")
        run_python("scripts/convert_trading_records.py")
        if full_portfolio:
            run_python("scripts/update_fa.py", "--full")
        else:
            run_python("scripts/update_fa.py")
    except subprocess.CalledProcessError as exc:
        print(f"(경고) 포트폴리오 계산 실패: {exc}")

    # 2. 인기 포스트 데이터 업데이트
    try:
        fetch_top_posts.main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            print("(경고) 인기글 데이터를 불러오지 못했습니다.")

    # 설정 파일에서 필요한 경로 및 값 로드
    temp_prefix = get_value("build.temp_prefix", "hugo_temp_")
    content_dir = get_path("content")
    public_dir = get_path("public")
    data_dir = get_path("data")

    # 빌드 전 메타데이터 디렉터리 정리
    clean_metadata_dirs([content_dir, data_dir, public_dir])

    # 3. 임시 디렉터리 내에서 빌드 수행
    with tempfile.TemporaryDirectory(prefix=temp_prefix) as temp_dir:
        temp_path = Path(temp_dir)
        # 원본 콘텐츠를 임시 디렉터리로 복사 (rsync 사용)
        run([
            "rsync",
            "-a",
            "--delete",
            f"{content_dir}/",
            f"{temp_path}/",
        ])

        # 4. 콘텐츠 전처리 스크립트 실행
        run_python("scripts/convert_to_webp.py", "--content-dir", str(content_dir))
        run_python("scripts/convert_wikilinks.py", "--content-dir", str(temp_path))
        run_python("scripts/replace_ad_marker.py", "--content-dir", str(temp_path))

        # 임시 디렉터리 내 메타데이터 정리
        clean_metadata_dirs([temp_path])

        # 5. Hugo 빌드 실행
        hugo_exe = get_value("hugo.executable", "hugo")
        hugo_args = get_value("hugo.args", []) or []
        hugo_cmd = [hugo_exe, *hugo_args, "--contentDir", str(temp_path)]
        run(hugo_cmd)

        # 6. 빌드 후처리
        run_python("scripts/convert_wikilinks.py", "--public-dir", str(public_dir))

    return public_dir


def sync_to_target(public_dir: Path, target: Path) -> None:
    """
    빌드된 `public` 디렉터리의 내용을 최종 목적지(`target`)로 동기화(rsync)합니다.
    - `hugo.yaml`의 `ssh_deploy.excludes` 설정에 따라 특정 파일/폴더를 제외합니다.
    """
    config = load_config()
    excludes = config.get("ssh_deploy", {}).get("excludes", []) or []
    target.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync",
        "-a",
        "--delete",
    ]
    for pattern in excludes:
        cmd.extend(["--exclude", pattern])
    cmd.extend([f"{public_dir}/", f"{target}/"])
    run(cmd)


def main() -> int:
    """스크립트의 메인 실행 함수."""
    # 명령줄 인자 파싱
    args = parse_args()
    # 설정 파일 로드
    load_config()

    # 사이트 빌드
    public_dir = build_site(args.full_portfolio)

    # 배포 대상 경로 결정
    target_path = args.target
    if target_path:
        sync_to_target(public_dir, target_path)
        print(f"배포 완료: {public_dir} → {target_path}")
    else:
        print(f"빌드 완료: {public_dir} (동기화 생략)")
    return 0


if __name__ == "__main__":
    # 스크립트가 직접 실행될 때 main 함수를 호출하고, 종료 코드를 시스템에 전달
    raise SystemExit(main())
