#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hugo.yaml 설정 파일을 사용하여 Hugo 사이트를 배포합니다."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import argparse

from scripts import fetch_top_posts
from scripts.config_utils import get_path, get_value, load_config

# 스크립트의 루트 디렉토리를 설정합니다.
ROOT = Path(__file__).resolve().parent
# 현재 사용 중인 파이썬 실행 파일의 경로를 가져옵니다.
PYTHON = sys.executable


def run(cmd: list[str], **kwargs) -> None:
    """주어진 명령어를 실행하고 표준 출력에 표시합니다."""
    print("$", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def run_python(script: str, *args: str) -> None:
    """루트 디렉토리에서 파이썬 스크립트를 실행합니다."""
    run([PYTHON, script, *args], cwd=ROOT)


def parse_args() -> argparse.Namespace:
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description="Hugo 사이트 배포")
    parser.add_argument(
        "--full",
        action="store_true",
        help="최신 월간 보고서 대신 전체 자산현황 보고서를 다시 생성합니다.",
    )
    return parser.parse_args()


def main() -> int:
    """메인 배포 로직을 실행합니다."""
    args = parse_args()
    config = load_config()

    try:
        # 자산현황 업데이트 스크립트를 실행합니다.
        run_python("scripts/convert_fa_md.py")
        run_python("scripts/convert_trading_records.py")
        if args.full:
            run_python("scripts/update_fa.py", "--full")
        else:
            run_python("scripts/update_fa.py")
    except Exception as exc:
        print(f"자산형환 계산 스크립트 실행 중 오류가 발생했습니다: {exc}")

    try:
        # 인기 글 데이터를 가져옵니다.
        fetch_top_posts.main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            print("인기 글 데이터를 불러오지 못했습니다.")

    # Hugo 빌드를 위한 임시 디렉토리를 생성합니다.
    temp_prefix = get_value("build.temp_prefix", "hugo_temp_")
    with tempfile.TemporaryDirectory(prefix=temp_prefix) as temp_dir:
        temp_path = Path(temp_dir)
        content_dir = get_path("content")
        public_dir = get_path("public")

        # 원본 콘텐츠를 임시 디렉토리로 복사합니다.
        run([
            "rsync",
            "-a",
            "--delete",
            f"{content_dir}/",
            f"{temp_path}/",
        ], cwd=ROOT)

        # 이미지와 링크를 변환하고 광고 마커를 교체합니다.
        run_python("scripts/convert_to_webp.py", "--content-dir", str(content_dir))
        run_python("scripts/convert_wikilinks.py", "--content-dir", str(temp_path))
        run_python("scripts/replace_ad_marker.py", "--content-dir", str(temp_path))

        # Hugo를 실행하여 사이트를 빌드합니다.
        hugo_exe = get_value("hugo.executable", "hugo")
        hugo_args = get_value("hugo.args", []) or []
        hugo_cmd = [hugo_exe, *hugo_args, "--contentDir", str(temp_path)]
        run(hugo_cmd, cwd=ROOT)

        # 빌드된 public 디렉토리의 위키링크를 변환합니다.
        run_python("scripts/convert_wikilinks.py", "--public-dir", str(public_dir))

    # SSH 배포 설정을 hugo.yaml에서 불러옵니다.
    ssh_conf = get_value("ssh_deploy", {}) or {}
    host = ssh_conf.get("host")  # SSH 호스트
    user = ssh_conf.get("user")  # SSH 사용자
    port = str(ssh_conf.get("port", 22))  # SSH 포트, 기본값 22
    drive_public = ssh_conf.get("drive_public")  # 원격 서버의 중간 저장 디렉토리
    web_public = ssh_conf.get("web_public")  # 원격 서버의 최종 웹 공개 디렉토리
    # rsync에서 제외할 파일/디렉토리 패턴 목록을 가져옵니다.
    excludes = ssh_conf.get("excludes", []) or []

    # 필수 설정값이 모두 있는지 확인합니다.
    if not all([host, user, drive_public, web_public]):
        print("ssh_deploy 설정이 부족합니다.")
        return 1

    # 원격 서버 주소를 생성합니다. (user@host)
    remote = f"{user}@{host}"

    # 원격 서버에 배포할 디렉토리들이 존재하는지 확인하고, 없으면 생성합니다.
    ssh_cmd = ["ssh", "-p", port, remote, f"mkdir -p '{drive_public}' '{web_public}'"]
    run(ssh_cmd)

    # 로컬 public 디렉토리의 내용을 원격 서버의 drive_public 디렉토리로 동기화(rsync)합니다.
    rsync_cmd = [
        "rsync",
        "-av",  # 아카이브 모드(-a)와 상세 정보 출력(-v)
        "--delete",  # 원본에 없는 파일은 대상에서 삭제
        # 제외할 패턴들을 '--exclude' 옵션으로 추가합니다.
        *[item for excl in excludes for item in ("--exclude", excl)],
        "-e",  # 원격 쉘(ssh)을 지정하고 포트 번호를 설정합니다.
        f"ssh -p {port}",
        f"{public_dir}/",  # 원본 디렉토리
        f"{remote}:{drive_public}/",  # 대상 디렉토리
    ]
    run(rsync_cmd)

    # 원격 서버의 드라이브 디렉토리에서 최종 웹 공개 디렉토리로 동기화합니다.
    # 이 단계는 원자적(atomic) 업데이트를 보장하기 위함일 수 있습니다.
    sync_remote_cmd = [
        "ssh",
        "-p",
        port,
        remote,
        "rsync -av --delete "
        # 제외할 패턴들을 쉘 환경에 맞게 '--exclude' 옵션으로 추가합니다.
        + " ".join(f"--exclude='{excl}'" for excl in excludes)
        + f" '{drive_public}/' '{web_public}/'",
    ]
    run(sync_remote_cmd)

    print("배포 완료!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
