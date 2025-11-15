#!/usr/bin/env python3
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

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    """명령 실행을 공통 처리한다."""
    print("$", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=cwd or ROOT)


def run_python(script: str, *args: str) -> None:
    """루트 디렉터리 기준으로 파이썬 스크립트를 실행한다."""
    run([PYTHON, script, *args], cwd=ROOT)


def clean_metadata_dirs(paths: list[Path]) -> None:
    """Synology에서 생성한 @eaDir 등의 메타데이터 폴더를 제거한다."""
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
    parser = argparse.ArgumentParser(description="NAS에서 Hugo 사이트 배포")
    parser.add_argument(
        "--full-portfolio",
        action="store_true",
        help="포트폴리오 전체 기간을 재계산합니다 (기본: 최신 월만 계산)",
    )
    parser.add_argument(
        "--target",
        type=Path,
        help="빌드된 public을 복사할 로컬 경로 (기본: hugo.yaml의 ssh_deploy.web_public)",
    )
    return parser.parse_args()


def build_site(full_portfolio: bool) -> Path:
    """포트폴리오 산출 + Hugo 빌드까지 수행하고 public 디렉터리를 반환한다."""
    try:
        if full_portfolio:
            run_python("scripts/update_fa.py", "--full")
        else:
            run_python("scripts/update_fa.py")
    except subprocess.CalledProcessError as exc:
        print(f"(경고) 포트폴리오 계산 실패: {exc}")

    try:
        fetch_top_posts.main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            print("(경고) 인기글 데이터를 불러오지 못했습니다.")

    temp_prefix = get_value("build.temp_prefix", "hugo_temp_")
    content_dir = get_path("content")
    public_dir = get_path("public")
    data_dir = get_path("data")

    clean_metadata_dirs([content_dir, data_dir, public_dir])

    with tempfile.TemporaryDirectory(prefix=temp_prefix) as temp_dir:
        temp_path = Path(temp_dir)
        run([
            "rsync",
            "-a",
            "--delete",
            f"{content_dir}/",
            f"{temp_path}/",
        ])

        run_python("scripts/convert_to_webp.py", "--content-dir", str(content_dir))
        run_python("scripts/convert_wikilinks.py", "--content-dir", str(temp_path))
        run_python("scripts/replace_ad_marker.py", "--content-dir", str(temp_path))

        clean_metadata_dirs([temp_path])

        hugo_exe = get_value("hugo.executable", "hugo")
        hugo_args = get_value("hugo.args", []) or []
        hugo_cmd = [hugo_exe, *hugo_args, "--contentDir", str(temp_path)]
        run(hugo_cmd)

        run_python("scripts/convert_wikilinks.py", "--public-dir", str(public_dir))

    return public_dir


def sync_to_target(public_dir: Path, target: Path) -> None:
    """public 디렉터리를 대상 경로로 rsync 한다."""
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
    args = parse_args()
    load_config()

    public_dir = build_site(args.full_portfolio)

    target_path = args.target
    if target_path is None:
        target_value = get_value("ssh_deploy.web_public")
        if not target_value:
            print("web_public 경로를 알 수 없습니다. --target 옵션을 지정하세요.")
            return 1
        target_path = Path(target_value)

    sync_to_target(public_dir, target_path)
    print(f"배포 완료: {public_dir} → {target_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
