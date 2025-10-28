#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SSH_PORT=45130
REMOTE_USER="vividian"
REMOTE_HOST="vividian.net"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
REMOTE_DRIVE_BASE="/var/services/homes/${REMOTE_USER}/Drive/Obsidian/blog"
REMOTE_PUBLIC="${REMOTE_DRIVE_BASE}/public"
REMOTE_WEB="/var/services/web/hugo"
SSH_OPTS=(-p "${SSH_PORT}")

echo "📊 인기 글 데이터 갱신 중..."
python3 scripts/fetch_top_posts.py || echo "(경고) 인기 글 데이터를 불러오지 못했습니다."

TEMP_CONTENT_DIR=$(mktemp -d)
cleanup() {
  rm -rf "$TEMP_CONTENT_DIR"
}
trap cleanup EXIT

rsync -a --delete content/ "$TEMP_CONTENT_DIR/"

echo "🖼  Obsidian 이미지 임베드 변환 중..."
python3 scripts/convert_obsidian_embeds.py --content-dir "$TEMP_CONTENT_DIR"

echo "📢 광고 마커 변환 중..."
python3 scripts/replace_ad_marker.py --content-dir "$TEMP_CONTENT_DIR"

echo "🚧 Hugo 빌드 시작..."
hugo --cleanDestinationDir --contentDir "$TEMP_CONTENT_DIR"
echo "✅ Hugo 빌드 완료"

echo "🔗 위키 링크 변환 중..."
python3 scripts/convert_wikilinks.py

echo "📁 원격 폴더 준비..."
ssh "${SSH_OPTS[@]}" "${REMOTE}" "mkdir -p '${REMOTE_PUBLIC}' '${REMOTE_WEB}'"

echo "📤 NAS Drive에 public 폴더 업로드..."
rsync -av --delete --exclude='**/@eaDir/**' -e "ssh -p ${SSH_PORT}" public/ "${REMOTE}:${REMOTE_PUBLIC}/"

echo "🚀 원격 서버에 배포 동기화..."
ssh "${SSH_OPTS[@]}" "${REMOTE}" "rsync -av --delete --exclude='**/@eaDir/**' '${REMOTE_PUBLIC}/' '${REMOTE_WEB}/'"

echo "🎉 배포 완료!"
