import subprocess
import yaml
from pathlib import Path

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
        process = subprocess.run(
            rsync_cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            encoding='utf-8'
        )
        print("STDOUT:", process.stdout)
        if process.stderr:
            print("STDERR:", process.stderr)
        print("동기화가 성공적으로 완료되었습니다.")
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
