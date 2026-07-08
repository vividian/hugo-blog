import re
import glob
import shutil
from pathlib import Path

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

# ── 설정 ──────────────────────────────────────────────────────────
blog_dir = Path("/Users/yjkim/Documents/vividian.net/Obsidian/blog")
vault_dir = blog_dir.parent  # Obsidian Vault 루트
content_dir = blog_dir / "content"
images_dir = content_dir / "images"

TARGET_WIDTH = 720
WEBP_QUALITY = 85

# 자동 생성 콘텐츠 등 이미지 처리를 건너뛸 디렉토리
EXCLUDE_DIRS = {"images", "fa", "search", "tags"}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif",
    ".bmp", ".tiff", ".tif", ".webp",
}


# ── 유틸리티 ──────────────────────────────────────────────────────
def slugify_hangul(text: str) -> str:
    """한글·영숫자만 남기고 공백은 밑줄로 변환합니다."""
    text = text.replace(" ", "_")
    text = re.sub(r"[^\w\u3131-\u3163\uac00-\ud7a3]", "", text)
    return text


def convert_to_webp(src_path: Path, dest_path: Path) -> None:
    """이미지를 WebP로 변환하고, 가로 720px 초과 시 리사이즈합니다."""
    if Image is None:
        print("  ⚠ Pillow 미설치 — 파일을 그대로 이동합니다.")
        shutil.move(str(src_path), str(dest_path))
        return

    try:
        with Image.open(src_path) as img:
            width, height = img.size
            if width > TARGET_WIDTH:
                new_height = int(height * TARGET_WIDTH / width)
                img = img.resize((TARGET_WIDTH, new_height), Image.LANCZOS)
            img.save(dest_path, "WEBP", quality=WEBP_QUALITY)
        src_path.unlink()  # 원본 삭제
    except Exception as e:
        print(f"  ⚠ 변환 실패 ({src_path.name}): {e}")
        shutil.move(str(src_path), str(dest_path))


def _is_under_images_dir(path: Path) -> bool:
    """path가 content/images/ 하위에 있는지 확인합니다."""
    try:
        path.resolve().relative_to(images_dir.resolve())
        return True
    except ValueError:
        return False


def resolve_image_path(filename: str, md_path: Path) -> Path | None:
    """이미지 파일의 실제 경로를 여러 위치에서 탐색합니다.
    이미 content/images/ 안에 올바른(.webp) 상태로 있거나, 이미지 확장자가 아니면 None을 반환합니다."""

    # 확장자가 이미지가 아닌 링크는 빠르게 건너뛰기
    ext = Path(filename).suffix.lower()
    if not ext or ext not in IMAGE_EXTENSIONS:
        return None

    # 이미 images/ 를 가리키는 링크이지만 확장자가 webp가 아닌 경우
    if "../images/" in filename or filename.startswith("images/"):
        if ext != ".webp":
            clean_fn = Path(filename).name
            target_path = images_dir / clean_fn
            if target_path.is_file():
                return target_path
        return None

    # 후보 경로를 순서대로 탐색
    candidates = [
        blog_dir / filename,        # 블로그 루트 기준 (Pasted 이미지 등)
        vault_dir / filename,       # Obsidian Vault 루트 기준 (Obsidian 붙여넣기 기본 경로 등)
        md_path.parent / filename,   # 마크다운 파일 기준 상대 경로
        content_dir / filename,      # content 디렉토리 기준
    ]

    for path in candidates:
        try:
            resolved = path.resolve()
        except (OSError, ValueError):
            continue
        if resolved.is_file() and resolved.suffix.lower() in IMAGE_EXTENSIONS:
            if _is_under_images_dir(resolved):
                return None  # 이미 images/ 에 있으면 건너뛰기
            return resolved

    # 파일명만으로 content 디렉토리 재귀 탐색
    basename = Path(filename).name
    for found in content_dir.rglob(basename):
        if found.is_file() and found.suffix.lower() in IMAGE_EXTENSIONS:
            if _is_under_images_dir(found):
                return None
            return found

    return None


# ── 메인 처리 ─────────────────────────────────────────────────────
def process_all_images() -> None:
    """content 디렉토리의 모든 마크다운에서 이미지를 찾아
    WebP 변환 후 content/images/ 로 이동하고 링크를 갱신합니다."""

    if not images_dir.exists():
        images_dir.mkdir(parents=True)

    md_files = glob.glob(str(content_dir / "**" / "*.md"), recursive=True)

    # 위키링크: [[filename]] 또는 [[filename|size]]
    wiki_pattern = r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]"
    # 표준 마크다운 이미지: ![alt](path)
    md_img_pattern = r"!\[([^\]]*)\]\(([^)\s]+)\)"

    total_processed = 0

    for file_path in md_files:
        md_path = Path(file_path)
        rel_path = md_path.relative_to(content_dir)

        # 카테고리 폴더 안에 있는 파일만 대상
        if len(rel_path.parts) < 2:
            continue

        category = rel_path.parts[0]

        # 제외 디렉토리 건너뛰기 (images, fa 등)
        if category in EXCLUDE_DIRS:
            continue

        # 접두어 생성: 카테고리_날짜_제목 형태
        name_without_ext = md_path.stem
        m_name = re.match(r"^(\d{6})\s+(.+)$", name_without_ext)
        if m_name:
            date_str = m_name.group(1)
            safe_name = slugify_hangul(m_name.group(2))
            prefix = f"{category}_{date_str}_{safe_name}"
        else:
            safe_name = slugify_hangul(name_without_ext)
            prefix = f"{category}_{safe_name}"

        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 위키링크와 마크다운 이미지 링크 모두 수집
        wiki_matches = [(m, "wiki") for m in re.finditer(wiki_pattern, content)]
        md_matches = [(m, "md") for m in re.finditer(md_img_pattern, content)]
        all_matches = wiki_matches + md_matches

        if not all_matches:
            continue

        file_mapping: dict[str, str] = {}  # old_filename -> new_filename

        # 이 prefix에 대해 기존 이미지 인덱스 최대값 조회
        existing_images = glob.glob(str(images_dir / f"{prefix}_*.webp"))
        max_index = 0
        for img_path in existing_images:
            basename = Path(img_path).stem
            parts = basename.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                if idx > max_index:
                    max_index = idx

        for m, link_type in all_matches:
            if link_type == "wiki":
                old_filename = m.group(1).strip()
            else:  # md
                old_filename = m.group(2).strip()

            # 이미지 실제 경로 탐색
            old_path = resolve_image_path(old_filename, md_path)
            if old_path is None:
                continue

            if old_filename not in file_mapping:
                if _is_under_images_dir(old_path):
                    new_filename = old_path.with_suffix(".webp").name
                else:
                    max_index += 1
                    new_ext = ".webp" if Image is not None else old_path.suffix.lower()
                    new_filename = f"{prefix}_{max_index}{new_ext}"
                
                new_path = images_dir / new_filename

                print(f"  처리: {old_path.name} -> {new_filename}")

                # 이미 webp인 파일은 리사이즈 필요 시에만 재인코딩
                if old_path.suffix.lower() == ".webp" and Image is not None:
                    with Image.open(old_path) as img:
                        if img.size[0] > TARGET_WIDTH:
                            convert_to_webp(old_path, new_path)
                        else:
                            shutil.move(str(old_path), str(new_path))
                elif Image is not None:
                    convert_to_webp(old_path, new_path)
                else:
                    shutil.move(str(old_path), str(new_path))

                file_mapping[old_filename] = new_filename
                total_processed += 1

        if not file_mapping:
            continue

        # ── 마크다운 내 이미지 링크 갱신 ──
        def replace_wiki_link(m):
            old_fn = m.group(1).strip()
            size_part = m.group(2)
            if old_fn in file_mapping:
                new_rel = f"../images/{file_mapping[old_fn]}"
                return f"[[{new_rel}|{size_part}]]" if size_part else f"[[{new_rel}]]"
            return m.group(0)

        def replace_md_img(m):
            alt_text = m.group(1)
            old_fn = m.group(2).strip()
            if old_fn in file_mapping:
                new_rel = f"../images/{file_mapping[old_fn]}"
                return f"![{alt_text}]({new_rel})"
            return m.group(0)

        new_content = re.sub(wiki_pattern, replace_wiki_link, content)
        new_content = re.sub(md_img_pattern, replace_md_img, new_content)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"  링크 갱신: {md_path.name}")

    if total_processed > 0:
        print(f"총 {total_processed}개 이미지 처리 완료.")
    else:
        print("처리할 이미지가 없습니다.")


if __name__ == "__main__":
    process_all_images()
