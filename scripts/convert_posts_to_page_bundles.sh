#!/usr/bin/env bash
# blog/content/posts 아래의 개별 Markdown 파일을 Hugo page bundle 구조로 변환한다.
# 각 파일 이름과 동일한 폴더를 만들고, 파일은 index.md로 이동한 뒤
# 본문에서 /wordpress/... 경로로 참조되는 이미지를 번들 폴더로 옮기고 링크도 갱신한다.

set -euo pipefail

POSTS_DIR="blog/content/posts"
STATIC_DIR="blog/static"

usage() {
  cat <<'EOF'
Usage: convert_posts_to_page_bundles.sh [--posts DIR] [--static DIR]

Options:
  --posts DIR   변환할 Markdown이 들어 있는 posts 디렉터리 (기본값: blog/content/posts)
  --static DIR  이미지가 위치한 static 루트 디렉터리 (기본값: blog/static)

주의:
  - 기존에 이미 page bundle 구조(폴더 + index.md)라면 건너뜁니다.
  - /wordpress/... 경로로 참조되는 이미지 파일을 번들 폴더로 이동시키므로,
    다른 문서에서 같은 이미지를 쓰고 있다면 링크가 깨질 수 있습니다.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --posts)
      POSTS_DIR=$2
      shift 2
      ;;
    --static)
      STATIC_DIR=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$POSTS_DIR" ]]; then
  echo "posts 디렉터리를 찾을 수 없습니다: $POSTS_DIR" >&2
  exit 1
fi

if [[ ! -d "$STATIC_DIR" ]]; then
  echo "static 디렉터리를 찾을 수 없습니다: $STATIC_DIR" >&2
  exit 1
fi

find "$POSTS_DIR" -maxdepth 1 -type f -name '*.md' -print0 |
while IFS= read -r -d '' md_file; do
  bundle_dir="${md_file%.md}"
  bundle_index="$bundle_dir/index.md"

  if [[ -d "$bundle_dir" ]]; then
    echo "이미 폴더가 존재하여 건너뜁니다: $bundle_dir" >&2
    continue
  fi

  echo "▶︎ 변환 시작: $md_file"
  mkdir -p "$bundle_dir"
  mv "$md_file" "$bundle_index"

  python3 - "$bundle_index" "$STATIC_DIR" <<'PYTHON'
import shutil
import sys
import re
from pathlib import Path

index_path = Path(sys.argv[1])
static_root = Path(sys.argv[2])
bundle_dir = index_path.parent

try:
    text = index_path.read_text(encoding="utf-8")
except UnicodeDecodeError:
    text = index_path.read_text(encoding="utf-8-sig")

pattern = re.compile(r"(/wordpress/[^\s\"'\)\]]+)")
paths = []
seen = set()
for match in pattern.finditer(text):
    url = match.group(1)
    if url not in seen:
        seen.add(url)
        paths.append(url)

if not paths:
    sys.exit(0)

replacements = {}

for url in paths:
    src = static_root / url.lstrip("/")
    if not src.exists():
        print(f"경고: 이미지 파일이 없음 -> {src}", file=sys.stderr)
        continue

    dest = bundle_dir / src.name
    if dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        counter = 1
        while dest.exists():
            counter += 1
            dest = bundle_dir / f"{stem}-{counter}{suffix}"

    print(f"  이미지 이동: {src} -> {dest}")
    shutil.move(str(src), dest)
    replacements[url] = dest.name

for old, new in replacements.items():
    text = text.replace(old, new)

if replacements:
    index_path.write_text(text, encoding="utf-8")
PYTHON

  echo "  완료: $bundle_dir"
done

echo "모든 파일 처리가 끝났습니다."
