#!/usr/bin/env python3
"""NAS scheduler helper.

Behavior:
1) Fetch root/content repositories.
2) If GitHub has new commits, hard-reset to origin and run full deploy.
3) If no new commits, run only update_fa.py and sync FA artifacts only.
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
    # Git 변경이 없을 때는 시세 업데이트 및 Plotly 대시보드만 초고속(수초)으로 갱신합니다.
    try:
        run([PYTHON, "scripts/update_fa_plotly.py"], cwd=ROOT, run_as=run_as)
    except subprocess.CalledProcessError as exc:
        print(f"(경고) update_fa_plotly.py 실행 실패: {exc}")
        print("(경고) 기존 최신 대시보드 파일을 유지하고 배포를 계속합니다.")


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

        # 세그먼트 렌더에 latest_fa.html이 포함된 경우 함께 반영합니다.
        rendered_latest = temp_path / "fa" / "latest_fa.html"
        if rendered_latest.is_file():
            latest_target = ROOT / "public" / "fa" / "latest_fa.html"
            latest_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rendered_latest, latest_target)
        return target


def sync_full_site(web_public: Path) -> None:
    public_dir = get_path("public")
    run(["rsync", "-rlptD", "--no-owner", "--no-group", "--chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r", "--delete", f"{public_dir}/", f"{web_public}/"], cwd=ROOT)


def sync_fa_artifacts(web_public: Path) -> None:
    fa_src_cfg = get_value("financial_assets.paths.static_dir", "content/fa")
    fa_src = Path(fa_src_cfg)
    if not fa_src.is_absolute():
        fa_src = (ROOT / fa_src).resolve()

    fa_dst = web_public / "fa"
    fa_dst.mkdir(parents=True, exist_ok=True)

    # 1. 핵심 대시보드 파일(HTML/JSON)을 직접 복사하여 권한 충돌 없이 즉시 반영
    for src_file, dst_name in [
        (ROOT / "content" / "fa" / "latest_fa.html", "latest_fa.html"),
        (ROOT / "generated" / "fa" / "latest_fa_fragment.html", "latest_fa_fragment.html"),
        (ROOT / "public" / "fa" / "index.html", "index.html"),
        (ROOT / "public" / "fa" / "latest_fa.html", "latest_fa.html"),
        (ROOT / "data" / "fa.json", "fa.json"),
    ]:
        if src_file.is_file():
            try:
                shutil.copy2(src_file, fa_dst / dst_name)
            except Exception as e:
                print(f"(참고) {dst_name} 직접 복사 건너뜀: {e}")

    # 2. 기타 산출물(webp/csv) 동기화 (권한 에러 시에도 프로세스 중단 방지)
    try:
        run(
            [
                "rsync",
                "-rlptD",
                "--no-owner",
                "--no-group",
                "--chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r",
                "--include",
                "*/",
                "--include",
                "*.html",
                "--include",
                "*.webp",
                "--include",
                "*.csv",
                "--include",
                "*.json",
                "--exclude",
                "*",
                f"{fa_src}/",
                f"{fa_dst}/",
            ],
            cwd=ROOT,
        )
    except Exception as exc:
        print(f"(참고) 일부 과거 이미지 webp rsync 건너뜀 (대시보드 HTML은 정상 반영됨): {exc}")


def apply_permissions(web_public: Path, owner: str) -> None:
    try:
        run(["chown", "-R", owner, str(web_public)], cwd=ROOT)
        run(["find", str(web_public), "-type", "d", "-exec", "chmod", "755", "{}", ";"], cwd=ROOT)
        run(["find", str(web_public), "-type", "f", "-exec", "chmod", "644", "{}", ";"], cwd=ROOT)
    except Exception as exc:
        print(f"(참고) 웹 디렉터리 권한 변경 건너뜀 (일반 사용자 실행): {exc}")


def apply_workspace_owner(owner: str) -> None:
    try:
        run(["chown", "-R", owner, str(ROOT)], cwd=ROOT)
    except Exception as exc:
        print(f"(참고) 작업 공간 권한 변경 건너뜀 (일반 사용자 실행): {exc}")


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
    parser.add_argument("--owner", default="vividian:http", help="Final web directory owner")
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
    else:
        print("원격 변경 없음: update_fa + FA 산출물만 동기화합니다.")
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
