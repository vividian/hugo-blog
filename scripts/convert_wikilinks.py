#!/usr/bin/env python3
"""Convert Obsidian-style wiki links ([[...]] / ![[...]]) inside Markdown or rendered HTML."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    from config_utils import get_path, get_value
except ImportError:
    from .config_utils import get_path, get_value

CONTENT_DIR = get_path("content")
PERMALINKS = get_value("wikilinks.permalinks", {}) or {}

WIKILINK_PATTERN = re.compile(r"(!?)\[\[([^\]]+)\]\]")
YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "www.youtu.be",
}
YOUTUBE_ID_PATTERN = re.compile(r"^[\w\-]{11}$")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"}
SIZE_PATTERN = re.compile(r"^(?P<width>\d+)(?:x(?P<height>\d+))?$", re.IGNORECASE)
FRONT_MATTER_PATTERN = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[\s]+", "-", value)
    value = re.sub(r"[^0-9a-z\-가-힣_]+", "", value)
    return value


def parse_front_matter(md_path: Path) -> dict[str, str]:
    text = md_path.read_text(encoding="utf-8")
    match = FRONT_MATTER_PATTERN.match(text)
    if not match:
        return {}
    front = match.group(0)
    result: dict[str, str] = {}
    for key in ("title", "slug", "url"):
        pattern = re.compile(f"^{key}:\\s*[\"']?([^\"'\\n]+)", re.MULTILINE)
        field = pattern.search(front)
        if field:
            result[key] = field.group(1).strip()
    return result


def build_mapping(content_dir: Path) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for md_path in content_dir.rglob("index.md"):
        rel_parts = md_path.relative_to(content_dir).parts
        if len(rel_parts) < 2:
            continue
        section = rel_parts[0]
        folder_name = rel_parts[-2]
        fm = parse_front_matter(md_path)
        slug = fm.get("slug") or slugify(folder_name)
        url = fm.get("url")
        if not url:
            pattern = PERMALINKS.get(section)
            if pattern:
                url = pattern.replace(":slug", slug)
            else:
                url = f"/{section}/{slug}/"
        keys = {
            folder_name.strip(): url,
            folder_name.strip().lower(): url,
            slug: url,
        }
        title = fm.get("title")
        if title:
            keys[title.strip()] = url
            keys[title.strip().lower()] = url
        entry = {"url": url, "title": fm.get("title") or folder_name}
        for key in keys:
            if key and key not in mapping:
                mapping[key] = entry
    return mapping


def find_entry(mapping: dict[str, dict[str, str]], target: str) -> dict[str, str] | None:
    if target in mapping:
        return mapping[target]
    lowered = target.lower()
    if lowered in mapping:
        return mapping[lowered]
    slugged = slugify(target)
    if slugged in mapping:
        return mapping[slugged]
    return None


def render_youtube_embed(video_id: str, label: str | None, width: str | None, height: str | None) -> str:
    title_attr = html.escape(label or "YouTube video")
    embed_src = html.escape(f"https://www.youtube.com/embed/{video_id}")
    width_attr = f' width="{width}"' if width else ""
    height_attr = ""
    if height:
        height_attr = f' height="{height}"'
    elif width:
        try:
            height_val = max(1, round(int(width) * 9 / 16))
            height_attr = f' height="{height_val}"'
        except ValueError:
            pass
    return (
        '<div class="video-embed youtube">'
        f'<iframe src="{embed_src}"{width_attr}{height_attr} '
        f'title="{title_attr}" frameborder="0" '
        'allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" '
        'allowfullscreen></iframe>'
        "</div>"
    )


def render_image(src: str, label: str | None, width: str | None, height: str | None) -> str:
    alt_text = label or Path(src).stem
    attrs = [
        f'src="{html.escape(src, quote=True)}"',
        f'alt="{html.escape(alt_text)}"',
    ]
    if width:
        attrs.append(f' width="{width}"')
    if height:
        attrs.append(f' height="{height}"')
    return f"<img {' '.join(attrs)} />"


def extract_youtube_id(url: str) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path

    if host not in YOUTUBE_HOSTS:
        return None

    if host.endswith("youtu.be"):
        return path.lstrip("/").split("/", 1)[0]
    if path.startswith("/watch"):
        return parse_qs(parsed.query).get("v", [None])[0]
    if path.startswith("/embed/"):
        parts = path.split("/")
        if len(parts) > 2:
            return parts[2]
    if path.startswith("/shorts/"):
        parts = path.split("/")
        if len(parts) > 2:
            return parts[2]
    return None


def build_replacement(match: re.Match[str], mapping: dict[str, dict[str, str]], allow_embed: bool) -> tuple[str, bool]:
    is_embed = match.group(1) == "!"
    raw = match.group(2)

    parts = [p.strip() for p in raw.split("|")]
    target = parts[0]
    extras = parts[1:]

    size_width = size_height = None
    if extras:
        size_match = SIZE_PATTERN.match(extras[-1])
        if size_match:
            size_width = size_match.group("width")
            size_height = size_match.group("height")
            extras = extras[:-1]

    custom_label = "|".join(extras) if extras else None

    target_is_url = target.startswith(("http://", "https://"))
    youtube_id = extract_youtube_id(target) if target_is_url else None

    if is_embed:
        if not allow_embed:
            return match.group(0), False
        if youtube_id and YOUTUBE_ID_PATTERN.match(youtube_id):
            return render_youtube_embed(youtube_id, custom_label or target, size_width, size_height), True
        return render_image(target, custom_label, size_width, size_height), True

    if youtube_id and YOUTUBE_ID_PATTERN.match(youtube_id):
        if not allow_embed:
            return match.group(0), False
        return render_youtube_embed(youtube_id, custom_label or target, size_width, size_height), True

    entry = find_entry(mapping, target)
    label = custom_label or (entry.get("title") if entry else target)

    if entry:
        return f'<a href="{html.escape(entry["url"])}">{html.escape(label)}</a>', True
    if target_is_url:
        return f'<a href="{html.escape(target, quote=True)}">{html.escape(custom_label or target)}</a>', True
    return html.escape(label), True


def convert_html(public_dir: Path, mapping: dict[str, dict[str, str]]) -> int:
    updated = 0
    for html_file in public_dir.rglob("*.html"):
        text = html_file.read_text(encoding="utf-8")
        body_match = re.search(r"(<body[^>]*>)(.*)(</body>)", text, flags=re.IGNORECASE | re.DOTALL)
        if not body_match:
            continue
        body_start = body_match.start(2)
        body_end = body_match.end(2)

        def repl(mt: re.Match[str]) -> str:
            allow = body_start <= mt.start() < body_end
            replacement, changed = build_replacement(mt, mapping, allow_embed=allow)
            if changed:
                repl.changed = True  # type: ignore[attr-defined]
            return replacement

        repl.changed = False  # type: ignore[attr-defined]
        new_text = WIKILINK_PATTERN.sub(repl, text)
        if repl.changed:
            updated += 1
            html_file.write_text(new_text, encoding="utf-8")
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content-dir", type=Path, help="Directory containing Markdown files")
    parser.add_argument("--public-dir", type=Path, help="Directory containing rendered HTML files")
    args = parser.parse_args()

    if args.content_dir and not args.public_dir:
        print("Markdown 파일은 수정하지 않습니다. --public-dir 옵션을 사용해 주세요.")
        return 0

    public_dir = args.public_dir.resolve() if args.public_dir else get_path("public")
    if not public_dir.is_dir():
        print(f"{public_dir} 디렉터리를 찾을 수 없습니다")
        return 1

    mapping = build_mapping(CONTENT_DIR)
    if not mapping:
        print("위키 링크 매핑 정보가 없습니다")

    converted = convert_html(public_dir, mapping)
    print(f"위키 링크 변환 완료 (HTML): {converted}개 파일")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
