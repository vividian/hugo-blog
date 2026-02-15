#!/usr/bin/env python3
"""NAS scheduler helper.

Behavior:
1) Fetch root/content repositories.
2) If GitHub has new commits, hard-reset to origin and run full deploy.
3) If no new commits, refresh FA outputs and sync FA artifacts only.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from config_utils import get_path, get_value
except ImportError:
    from .config_utils import get_path, get_value


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
PYTHON = str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable))


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = False,
    run_as: str = "",
) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(str(part) for part in cmd))
    effective_cwd = cwd or ROOT
    exec_cmd = cmd
    if run_as:
        shell_cmd = f"cd {shlex.quote(str(effective_cwd))} && " + " ".join(
            shlex.quote(str(part)) for part in cmd
        )
        exec_cmd = ["su", "-s", "/bin/bash", run_as, "-c", shell_cmd]
        effective_cwd = ROOT
    return subprocess.run(
        exec_cmd,
        check=True,
        cwd=effective_cwd,
        text=True,
        encoding="utf-8",
        capture_output=capture,
    )


def git_head(repo_dir: Path, ref: str, run_as: str) -> str:
    result = run(["git", "rev-parse", ref], cwd=repo_dir, capture=True, run_as=run_as)
    return result.stdout.strip()


def fetch(repo_dir: Path, remote: str, run_as: str) -> None:
    run(["git", "fetch", remote], cwd=repo_dir, run_as=run_as)


def has_remote_update(repo_dir: Path, remote: str, branch: str, run_as: str) -> bool:
    local_head = git_head(repo_dir, "HEAD", run_as)
    remote_head = git_head(repo_dir, f"{remote}/{branch}", run_as)
    changed = local_head != remote_head
    status = "changed" if changed else "unchanged"
    print(f"[git] {repo_dir}: {status} ({local_head[:7]} -> {remote_head[:7]})")
    return changed


def hard_reset(repo_dir: Path, remote: str, branch: str, run_as: str) -> None:
    run(["git", "reset", "--hard", f"{remote}/{branch}"], cwd=repo_dir, run_as=run_as)


def run_full_deploy(
    root_remote: str,
    root_branch: str,
    content_remote: str,
    content_branch: str,
    run_as: str,
) -> None:
    content_repo = get_path("content")
    public_dir = get_path("public")

    hard_reset(ROOT, root_remote, root_branch, run_as)
    hard_reset(content_repo, content_remote, content_branch, run_as)

    if public_dir.exists():
        print(f"public 디렉터리 삭제: {public_dir}")
        shutil.rmtree(public_dir)

    run([PYTHON, "deploy.py", "--nas"], cwd=ROOT, run_as=run_as)


def run_fa_refresh(run_as: str) -> None:
    # Git 변경이 없을 때도 FA 이미지/HTML 리포트를 함께 갱신합니다.
    # HTML은 Hugo static 경로에 생성해 /fa/latest_fa.html 로 항상 서빙되게 합니다.
    static_fa_dir = ROOT / "static" / "fa"
    static_fa_dir.mkdir(parents=True, exist_ok=True)
    latest_fa_html = static_fa_dir / "latest_fa.html"
    run([PYTHON, "scripts/update_fa.py"], cwd=ROOT, run_as=run_as)
    run(
        [
            PYTHON,
            "scripts/update_fa_plotly.py",
            "--output",
            str(latest_fa_html),
        ],
        cwd=ROOT,
        run_as=run_as,
    )


def render_fa_index(run_as: str) -> Path:
    hugo_exe = get_value("hugo.executable", "hugo")
    hugo_args = get_value("hugo.args", []) or []
    with tempfile.TemporaryDirectory(prefix="hugo_fa_") as temp_dir:
        temp_path = Path(temp_dir)
        # root 스케줄러에서 실행될 때도 --run-as 사용자가 destination에 쓸 수 있어야 한다.
        if run_as:
            run(["chown", run_as, str(temp_path)], cwd=ROOT)
            run(["chmod", "755", str(temp_path)], cwd=ROOT)
        run(
            [
                hugo_exe,
                *hugo_args,
                "--config",
                "hugo.yaml,config/config.yaml",
                "--renderSegments",
                "fa",
                "--noTimes",
                "--noChmod",
                "--destination",
                str(temp_path),
            ],
            cwd=ROOT,
            run_as=run_as,
        )
        rendered = temp_path / "fa" / "index.html"
        if not rendered.is_file():
            raise FileNotFoundError(f"렌더된 fa/index.html을 찾을 수 없습니다: {rendered}")
        target = ROOT / "public" / "fa" / "index.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rendered, target)
        return target


def sync_full_site(web_public: Path) -> None:
    public_dir = get_path("public")
    run(["rsync", "-a", "--delete", f"{public_dir}/", f"{web_public}/"], cwd=ROOT)


def sync_fa_artifacts(web_public: Path) -> None:
    fa_src_cfg = get_value("financial_assets.paths.static_dir", "content/fa")
    fa_src = Path(fa_src_cfg)
    if not fa_src.is_absolute():
        fa_src = (ROOT / fa_src).resolve()

    fa_dst = web_public / "fa"
    fa_dst.mkdir(parents=True, exist_ok=True)

    # 최신 시세 반영에 필요한 산출물(webp/csv/json/html)만 복사합니다.
    run(
        [
            "rsync",
            "-a",
            "--delete",
            "--include",
            "*/",
            "--include",
            "*.webp",
            "--include",
            "*.csv",
            "--include",
            "*.json",
            "--include",
            "*.html",
            "--exclude",
            "*",
            f"{fa_src}/",
            f"{fa_dst}/",
        ],
        cwd=ROOT,
    )
    static_latest_html = ROOT / "static" / "fa" / "latest_fa.html"
    if static_latest_html.is_file():
        # 외부 스케줄러가 public -> web 동기화를 수행해도 파일이 사라지지 않도록 public에도 맞춰둡니다.
        public_latest = get_path("public") / "fa" / "latest_fa.html"
        public_latest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(static_latest_html, public_latest)
        run(["rsync", "-a", str(static_latest_html), str(fa_dst / "latest_fa.html")], cwd=ROOT)
    rendered_index = get_path("public") / "fa" / "index.html"
    if rendered_index.is_file():
        run(["rsync", "-a", str(rendered_index), str(fa_dst / "index.html")], cwd=ROOT)


def apply_permissions(web_public: Path, owner: str) -> None:
    run(["chown", "-R", owner, str(web_public)], cwd=ROOT)
    run(["find", str(web_public), "-type", "d", "-exec", "chmod", "755", "{}", ";"], cwd=ROOT)
    run(["find", str(web_public), "-type", "f", "-exec", "chmod", "644", "{}", ";"], cwd=ROOT)


def apply_workspace_owner(owner: str) -> None:
    run(["chown", "-R", owner, str(ROOT)], cwd=ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-remote", default="origin")
    parser.add_argument("--root-branch", default="main")
    parser.add_argument("--content-remote", default="origin")
    parser.add_argument("--content-branch", default="main")
    parser.add_argument(
        "--run-as",
        default="",
        help="Optional user for workspace operations (git/build/update_fa)",
    )
    parser.add_argument("--owner", default="http:http", help="Final web directory owner")
    parser.add_argument("--skip-permissions", action="store_true", help="Skip chown/chmod")
    parser.add_argument(
        "--workspace-owner",
        default="",
        help="Optional owner for blog workspace (ex: vividian:users)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    content_repo = get_path("content")
    web_public_cfg = get_value("ssh_deploy.web_public")
    if not web_public_cfg:
        print("ssh_deploy.web_public 설정이 없습니다.")
        return 1
    web_public = Path(web_public_cfg)

    fetch(ROOT, args.root_remote, args.run_as)
    fetch(content_repo, args.content_remote, args.run_as)

    root_changed = has_remote_update(ROOT, args.root_remote, args.root_branch, args.run_as)
    content_changed = has_remote_update(content_repo, args.content_remote, args.content_branch, args.run_as)
    has_changes = root_changed or content_changed

    if has_changes:
        print("원격 변경 감지: 전체 배포를 수행합니다.")
        run_full_deploy(
            args.root_remote,
            args.root_branch,
            args.content_remote,
            args.content_branch,
            args.run_as,
        )
        sync_full_site(web_public)
        # Hugo full build에서는 content/fa/latest_fa.html이 누락될 수 있어 FA 산출물을 후동기화합니다.
        sync_fa_artifacts(web_public)
    else:
        print("원격 변경 없음: update_fa/update_fa_plotly + FA 산출물만 동기화합니다.")
        run_fa_refresh(args.run_as)
        render_fa_index(args.run_as)
        sync_fa_artifacts(web_public)

    if not args.skip_permissions:
        apply_permissions(web_public, args.owner)
    if args.workspace_owner:
        apply_workspace_owner(args.workspace_owner)

    print("완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
