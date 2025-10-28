#!/usr/bin/env bash
# Rename percent-encoded Markdown filenames in blog/content/posts to Hangul.
# Run with --dry-run first to confirm the planned renames.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: rename_posts_to_hangul.sh [--target DIR] [--recursive] [--dry-run]

Options:
  --target DIR   Directory containing Markdown posts (default: blog/content/posts)
  --recursive    Traverse subdirectories under the target
  --dry-run      Preview the planned renames without changing files
EOF
}

TARGET="blog/content/posts"
RECURSIVE=0
DRY_RUN=0

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
  echo "Target directory not found: $TARGET" >&2
  exit 1
fi

mapfile -t FILES < <(
  if [[ $RECURSIVE -eq 1 ]]; then
    find "$TARGET" -type f -name '*.md' -print0 | xargs -0 -n1 printf '%s\n'
  else
    find "$TARGET" -maxdepth 1 -type f -name '*.md' -print0 | xargs -0 -n1 printf '%s\n'
  fi
)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No Markdown files found."
  exit 0
fi

declare -a RENAMES=()
declare -a WARNINGS=()

for SRC in "${FILES[@]}"; do
  BASENAME=$(basename "$SRC")
  DECODED=$(python3 - <<'PYTHON' "$BASENAME"
import sys
from urllib.parse import unquote
print(unquote(sys.argv[1]))
PYTHON
)

  if [[ "$DECODED" == "$BASENAME" ]]; then
    continue
  fi

  DEST_DIR=$(dirname "$SRC")
  DEST="$DEST_DIR/$DECODED"

  if [[ -e "$DEST" && "$DEST" != "$SRC" ]]; then
    WARNINGS+=("skip (already exists): $SRC -> $DEST")
    continue
  fi

  if printf '%s\0' "${RENAMES[@]}" | grep -Fzx "$DEST"; then
    WARNINGS+=("skip (duplicate target): $SRC -> $DEST")
    continue
  fi

  RENAMES+=("$SRC" "$DEST")
done

if [[ ${#WARNINGS[@]} -gt 0 ]]; then
  echo "Warnings:"
  for WARNING in "${WARNINGS[@]}"; do
    echo "  $WARNING"
  done
  echo
fi

if [[ ${#RENAMES[@]} -eq 0 ]]; then
  echo "No safe renames to apply."
  exit 1
fi

echo "Planned renames:"
for ((i = 0; i < ${#RENAMES[@]}; i += 2)); do
  SRC=${RENAMES[i]}
  DEST=${RENAMES[i+1]}
  REL_SRC=${SRC#"$TARGET"/}
  REL_DEST=${DEST#"$TARGET"/}
  echo "  $REL_SRC -> $REL_DEST"
done

if [[ $DRY_RUN -eq 1 ]]; then
  echo
  echo "Dry-run complete. Re-run without --dry-run to apply changes."
  exit 0
fi

for ((i = 0; i < ${#RENAMES[@]}; i += 2)); do
  SRC=${RENAMES[i]}
  DEST=${RENAMES[i+1]}
  mv "$SRC" "$DEST"
done

echo
echo "Renaming complete."
