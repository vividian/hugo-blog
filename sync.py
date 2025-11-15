import subprocess
import yaml
from pathlib import Path


def run_process(cmd, *, cwd=None, capture=False):
    return subprocess.run(
        cmd,
        check=True,
        cwd=cwd,
        text=True,
        capture_output=capture,
        encoding="utf-8",
    )


def git_push_blog(root_dir: Path) -> None:
    print("blog 저장소 변경 사항 푸시 중...")
    run_process(["git", "push"], cwd=root_dir)
    print("blog 저장소 푸시 완료.")


def git_sync_content(content_dir: Path) -> None:
    if not content_dir.exists():
        print(f"경고: 콘텐츠 디렉터리를 찾을 수 없습니다: {content_dir}")
        return
    print("content 저장소 상태 확인 중...")
    status = run_process(["git", "status", "--porcelain"], cwd=content_dir, capture=True)
    if status.stdout.strip():
        print("변경 사항을 발견했습니다. 커밋을 생성합니다.")
        run_process(["git", "add", "-A"], cwd=content_dir)
        run_process(["git", "commit", "-m", "게시글 업데이트"], cwd=content_dir)
    else:
        print("커밋할 변경 사항이 없어 커밋을 생략합니다.")
    print("content 저장소 푸시 중...")
    run_process(["git", "push"], cwd=content_dir)
    print("content 저장소 푸시 완료.")

def main():
    """
    로컬 config 폴더를 원격 NAS의 config 폴더와 동기화합니다.
    config/config.yaml 파일에서 SSH 접속 정보를 읽어 사용합니다.
    """
    # 블로그 루트 디렉토리 설정
    root_dir = Path(__file__).parent.resolve()
    config_path = root_dir / "config" / "config.yaml"
    local_config_dir = root_dir / "config/"  # rsync를 위해 끝에 슬래시 추가

    # 설정 파일 로드
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"오류: 설정 파일을 찾을 수 없습니다: {config_path}")
        return
    except Exception as e:
        print(f"오류: 설정 파일을 읽는 중 문제가 발생했습니다: {e}")
        return

    # SSH 설정 정보 추출
    ssh_config = config.get("ssh_deploy", {})
    host = ssh_config.get("host")
    user = ssh_config.get("user")
    port = ssh_config.get("port")
    drive_public = ssh_config.get("drive_public")

    if not all([host, user, port, drive_public]):
        print("오류: config.yaml 파일에 SSH 배포 설정이 완전하지 않습니다.")
        print(" (host, user, port, drive_public)")
        return

    # 원격지 경로 설정 (drive_public 경로를 기반으로)
    remote_blog_root = Path(drive_public).parent
    remote_config_path = remote_blog_root / "config"

    # rsync 명령어 구성
    rsync_cmd = [
        "rsync",
        "-avz",
        "-e",
        f"ssh -p {port}",
        "--delete",
        str(local_config_dir),
        f"{user}@{host}:{remote_config_path}"
    ]

    print("NAS와 config 폴더 동기화를 시작합니다...")
    print(f"실행 명령어: {' '.join(rsync_cmd)}")

    # rsync 실행
    try:
        process = run_process(
            rsync_cmd,
            capture=True,
        )
        if process.stdout:
            print("STDOUT:", process.stdout)
        if process.stderr:
            print("STDERR:", process.stderr)
        print("동기화가 성공적으로 완료되었습니다.")
        git_push_blog(root_dir)
        git_sync_content(root_dir / "content")
    except subprocess.CalledProcessError as e:
        print(f"오류: 동기화 중 오류가 발생했습니다 (Exit Code: {e.returncode}).")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
    except FileNotFoundError:
        print("오류: 'rsync' 명령을 찾을 수 없습니다. rsync가 설치되어 있고 PATH에 등록되어 있는지 확인해주세요.")
    except Exception as e:
        print(f"알 수 없는 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()
