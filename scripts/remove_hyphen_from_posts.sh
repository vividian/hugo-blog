#!/usr/bin/env bash
# Markdown 파일 이름에서 하이픈(-)을 공백으로 바꾸는 스크립트.
# 기본 대상: blog/content/posts. 먼저 --dry-run으로 실행해 변경 사항을 확인하세요.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: remove_hyphen_from_posts.sh [--target DIR] [--recursive] [--dry-run]

Options:
  --target DIR   처리할 디렉터리 (기본값: blog/content/posts)
  --recursive    하위 디렉터리까지 모두 처리
  --dry-run      실제 변경 없이 예정된 작업만 출력 (기본값)
  --no-dry-run   실제로 이름을 변경
EOF
}

TARGET="blog/content/posts"
RECURSIVE=0
DRY_RUN=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET=$2
      shift 2
      ;;
    --recursive)
      RECURSIVE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-dry-run)
      DRY_RUN=0
      shift
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

if [[ ! -d "$TARGET" ]]; then
  echo "대상 디렉터리를 찾을 수 없습니다: $TARGET" >&2
  exit 1
fi

find_args=(-type f -name '*.md')
if [[ $RECURSIVE -eq 0 ]]; then
  find_args=(-maxdepth 1 "${find_args[@]}")
fi

declare -a renames=()
declare -a warnings=()

while IFS= read -r -d '' src; do
  basename=${src##*/}
  stem=${basename%.md}
  new_stem=${stem//-/ }

  # 연속 공백을 하나로 줄이고 앞뒤 공백 제거
  while [[ "$new_stem" == *"  "* ]]; do
    new_stem=${new_stem//  / }
  done
  while [[ "$new_stem" == " "* ]]; do
    new_stem=${new_stem# }
  done
  while [[ "$new_stem" == *" " ]]; do
    new_stem=${new_stem% }
  done

  new_basename="$new_stem.md"

  if [[ "$new_basename" == "$basename" ]]; then
    continue
  fi

  dest_dir=${src%/*}

  if [[ -z "$new_stem" ]]; then
    warnings+=("skip (empty name): $src -> $dest_dir/.md")
    continue
  fi

  dest="$dest_dir/$new_basename"

  if [[ -e "$dest" && "$dest" != "$src" ]]; then
    warnings+=("skip (already exists): $src -> $dest")
    continue
  fi

  duplicate=0
  for ((i=0; i<${#renames[@]}; i+=2)); do
    if [[ "${renames[i+1]}" == "$dest" && "${renames[i]}" != "$src" ]]; then
      duplicate=1
      break
    fi
  done
  if (( duplicate )); then
    warnings+=("skip (duplicate target): $src -> $dest")
    continue
  fi

  renames+=("$src" "$dest")
done < <(find "$TARGET" "${find_args[@]}" -print0)

if (( ${#renames[@]} == 0 )); then
  echo "하이픈을 제거할 Markdown 파일이 없습니다."
  exit 0
fi

if (( ${#warnings[@]} > 0 )); then
  echo "경고:"
  for warning in "${warnings[@]}"; do
    echo "  $warning"
  done
  echo
fi

echo "변경 예정 목록:"
for ((i=0; i<${#renames[@]}; i+=2)); do
  src=${renames[i]}
  dest=${renames[i+1]}
  rel_src=${src#"$TARGET"/}
  rel_dest=${dest#"$TARGET"/}
  echo "  $rel_src -> $rel_dest"
done

if (( DRY_RUN )); then
  echo
  echo "--dry-run 상태입니다. 실제로 이름을 바꾸려면 --no-dry-run 옵션을 추가하세요."
  exit 0
fi

echo
for ((i=0; i<${#renames[@]}; i+=2)); do
  mv "${renames[i]}" "${renames[i+1]}"
done

echo "이름 변경이 완료되었습니다."
