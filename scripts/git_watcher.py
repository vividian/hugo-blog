#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1분 주기 Git ls-remote 초경량 변경 감지 및 자동 빌드 왓처 (Watcher)
- 네트워크 패킷과 CPU 사용을 최소화하기 위해 'git ls-remote'로 원격 커밋 해시만 60초마다 확인합니다.
- 새 커밋이 감지되었을 때만 'git pull' 및 'deploy.py --nas' (Hugo 빌드)를 수행합니다.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTENT_DIR = ROOT / "content"
INTERVAL = int(os.environ.get("WATCHER_INTERVAL", 60))
PYTHON = sys.executable


def get_local_head(repo_dir: Path) -> str:
    """로컬 저장소의 현재 HEAD 커밋 해시 반환"""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception as e:
        return ""


def get_remote_head(repo_dir: Path, branch: str = "main") -> str | None:
    """git ls-remote를 통해 원격 브랜치의 최신 커밋 해시만 초경량으로 조회"""
    try:
        res = subprocess.run(
            ["git", "ls-remote", "origin", f"refs/heads/{branch}"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if res.returncode == 0 and res.stdout.strip():
            # 출력 형식: <commit_hash>\trefs/heads/<branch>
            return res.stdout.split()[0].strip()
        elif res.returncode != 0:
            err_msg = res.stderr.strip()
            if "Permission denied" in err_msg or "Could not read from remote" in err_msg:
                print(f"[{now()}] ⚠️ Git 인증 오류: GitHub 접근 권한(SSH 키 또는 PAT) 확인이 필요합니다.")
            elif "Could not resolve hostname" in err_msg:
                print(f"[{now()}] ⚠️ Git 호스트 해석 오류: ~/.ssh/config 설정을 확인하세요.")
            else:
                print(f"[{now()}] ⚠️ git ls-remote 실패: {err_msg}")
    except subprocess.TimeoutExpired:
        print(f"[{now()}] ⚠️ git ls-remote 시간 초과 (네트워크 확인 필요)")
    except Exception as e:
        print(f"[{now()}] ⚠️ git ls-remote 에러: {e}")
    return None


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sync_and_build(repo_dir: Path, branch: str = "main") -> bool:
    """원격 저장소의 변경사항을 가져와서 Hugo 사이트를 빌드합니다."""
    print(f"[{now()}] 🚀 새 글(커밋) 감지! 최신 변경 사항 동기화 및 사이트 빌드 시작...")
    try:
        # 1. 깃 최신화 (hard reset으로 로컬 변경 충돌 방지)
        subprocess.run(["git", "fetch", "origin", branch], cwd=repo_dir, check=True)
        subprocess.run(["git", "reset", "--hard", f"origin/{branch}"], cwd=repo_dir, check=True)
        print(f"[{now()}] ✅ Git 동기화 완료: origin/{branch}")

        # 2. Hugo 사이트 배포 빌드 실행
        print(f"[{now()}] 🔨 Hugo 사이트 빌드 실행 중...")
        build_res = subprocess.run(
            [PYTHON, "deploy.py", "--nas"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if build_res.returncode == 0:
            print(f"[{now()}] 🎉 블로그 배포 완료! 최신 글이 반영되었습니다.")
            return True
        else:
            print(f"[{now()}] ❌ 빌드 실패:\n{build_res.stderr}")
    except Exception as e:
        print(f"[{now()}] ❌ 동기화/빌드 중 예외 발생: {e}")
    return False


def watch() -> None:
    print(f"[{now()}] === Hugo Git 왓처(Watcher) 시작 ===")
    print(f"감시 대상: {CONTENT_DIR}")
    print(f"감시 주기: {INTERVAL}초 (초경량 git ls-remote 방식)")

    # safe.directory 등록
    subprocess.run(["git", "config", "--global", "--add", "safe.directory", "*"], check=False)

    last_checked_head = get_local_head(CONTENT_DIR)
    print(f"[{now()}] 현재 로컬 HEAD: {last_checked_head[:7] if last_checked_head else '없음'}")

    while True:
        time.sleep(INTERVAL)
        remote_head = get_remote_head(CONTENT_DIR)

        if not remote_head:
            continue

        local_head = get_local_head(CONTENT_DIR)
        if remote_head != local_head:
            print(f"[{now()}] [변경 감지] 로컬({local_head[:7]}) != 원격({remote_head[:7]})")
            success = sync_and_build(CONTENT_DIR)
            if success:
                last_checked_head = remote_head
        else:
            # 변경 없음: 불필요한 로그는 생략하고 대기 (원할 경우 디버그용 출력 가능)
            pass


if __name__ == "__main__":
    try:
        watch()
    except KeyboardInterrupt:
        print(f"\n[{now()}] 왓처를 종료합니다.")
