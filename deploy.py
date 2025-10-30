#!/usr/bin/env python3
"""Deploy the Hugo site using configuration from config.yaml."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from scripts import fetch_top_posts
from scripts.config_utils import get_path, get_value, load_config

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run(cmd: list[str], **kwargs) -> None:
    print("$", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def run_python(script: str, *args: str) -> None:
    run([PYTHON, script, *args], cwd=ROOT)


def main() -> int:
    config = load_config()

    try:
        fetch_top_posts.main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            print("(경고) 인기 글 데이터를 불러오지 못했습니다.")

    temp_prefix = get_value("build.temp_prefix", "hugo_temp_")
    with tempfile.TemporaryDirectory(prefix=temp_prefix) as temp_dir:
        temp_path = Path(temp_dir)
        content_dir = get_path("content")
        run([
            "rsync",
            "-a",
            "--delete",
            f"{content_dir}/",
            f"{temp_path}/",
        ], cwd=ROOT)

        run_python("scripts/convert_obsidian_embeds.py", "--content-dir", str(temp_path))
        run_python("scripts/replace_ad_marker.py", "--content-dir", str(temp_path))

        hugo_exe = get_value("hugo.executable", "hugo")
        hugo_args = get_value("hugo.args", []) or []
        hugo_cmd = [hugo_exe, *hugo_args, "--contentDir", str(temp_path)]
        run(hugo_cmd, cwd=ROOT)

    run_python("scripts/convert_wikilinks.py")

    ssh_conf = get_value("ssh_deploy", {}) or {}
    host = ssh_conf.get("host")
    user = ssh_conf.get("user")
    port = str(ssh_conf.get("port", 22))
    drive_public = ssh_conf.get("drive_public")
    web_public = ssh_conf.get("web_public")
    excludes = ssh_conf.get("excludes", []) or []

    if not all([host, user, drive_public, web_public]):
        print("ssh_deploy 설정이 부족합니다.")
        return 1

    remote = f"{user}@{host}"
    public_dir = get_path("public")

    ssh_cmd = ["ssh", "-p", port, remote, f"mkdir -p '{drive_public}' '{web_public}'"]
    run(ssh_cmd)

    rsync_cmd = [
        "rsync",
        "-a",
        "--delete",
        *[item for excl in excludes for item in ("--exclude", excl)],
        "-e",
        f"ssh -p {port}",
        f"{public_dir}/",
        f"{remote}:{drive_public}/",
    ]
    run(rsync_cmd)

    sync_remote_cmd = [
        "ssh",
        "-p",
        port,
        remote,
        "rsync -av --delete "
        + " ".join(f"--exclude='{excl}'" for excl in excludes)
        + f" '{drive_public}/' '{web_public}/'",
    ]
    run(sync_remote_cmd)

    print("배포 완료!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
