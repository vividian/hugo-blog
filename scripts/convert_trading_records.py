
import re
import csv
from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def _load_paths() -> tuple[Path, Path]:
    config = {}
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as fp:
            config = yaml.safe_load(fp) or {}
    paths = (config.get("financial_assets") or {}).get("paths") or {}
    csv_path = _resolve_path(paths.get("trading_records", "config/trading_records.csv"))
    md_path = _resolve_path(paths.get("trading_records_md", "content/fa/trading_records.md"))
    return csv_path, md_path


CSV_PATH, MD_PATH = _load_paths()

def convert_md_to_csv():
    """
    trading_records.md 파일의 마크다운 테이블을 trading_records.csv 파일로 변환합니다.
    """
    csv_file_path = CSV_PATH
    md_file_path = MD_PATH

    if not md_file_path.exists():
        print(f"Markdown 원본을 찾을 수 없습니다: {md_file_path}")
        return

    with open(md_file_path, 'r', encoding='utf-8') as md_file:
        content = md_file.read()

    # BOM (Byte Order Mark) 제거
    content = content.lstrip("\ufeff")

    # ---로 시작하고 끝나는 front matter 부분을 제거합니다.
    content = re.sub(r'^---[\s\S]*?---', '', content, flags=re.MULTILINE).strip()

    lines = content.split('\n')

    # 테이블의 헤더, 구분선, 데이터 행을 찾습니다.
    header_line = None
    data_lines = []
    found_table_blocks = []

    # 파일 내에서 유효한 테이블 블록(헤더, 구분선, 데이터)을 모두 찾습니다.
    for i, line in enumerate(lines):
        # 구분선인지 확인
        is_separator = '|' in line and '---' in line
        # 헤더 라인 후보인지 확인 (구분선 바로 윗줄)
        is_header_candidate = i > 0 and '|' in lines[i-1] and '---' not in lines[i-1]
        # 데이터 라인 후보인지 확인 (구분선 바로 아랫줄)
        is_data_candidate = i + 1 < len(lines) and '|' in lines[i+1]

        if is_separator and is_header_candidate and is_data_candidate:
            current_header = lines[i-1]
            current_data = []
            # 구분선 다음 줄부터 데이터 수집
            for j in range(i + 1, len(lines)):
                data_line = lines[j]
                if '|' in data_line:
                    current_data.append(data_line)
                else:
                    break  # 테이블 블록이 끝나면 중단
            found_table_blocks.append((current_header, current_data))

    # 찾은 테이블 블록 중 마지막 것을 사용합니다.
    if found_table_blocks:
        header_line, data_lines = found_table_blocks[-1]
    if not header_line:
        print("Markdown 테이블 헤더를 찾을 수 없습니다.")
        return

    # 헤더를 파싱합니다.
    header = [h.strip() for h in header_line.strip().strip('|').split('|')]

    # 데이터 행을 파싱합니다.
    data_rows = []
    for line in data_lines:
        # 각 셀의 앞뒤 공백을 제거합니다.
        cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
        if len(cells) == len(header):
            data_rows.append(cells)

    # CSV 파일에 씁니다.
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(data_rows)

    print(f"'{md_file_path}'의 테이블이 '{csv_file_path}'로 성공적으로 변환되었습니다.")

if __name__ == '__main__':
    convert_md_to_csv()
