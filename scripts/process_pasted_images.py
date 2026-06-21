import os
import re
import glob
from pathlib import Path
try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

blog_dir = Path("/Users/yjkim/Documents/vividian.net/Obsidian/blog")
content_dir = blog_dir / "content"
images_dir = content_dir / "images"

TARGET_WIDTH = 720
WEBP_QUALITY = 85

def slugify_hangul(text):
    text = text.replace(" ", "_")
    text = re.sub(r'[^\w\s\u3131-\u3163\uac00-\ud7a3]', '', text)
    return text

def convert_to_webp(src_path: Path, dest_path: Path):
    if Image is None:
        print("Pillow not installed. Moving file without conversion.")
        import shutil
        shutil.move(str(src_path), str(dest_path))
        return

    try:
        with Image.open(src_path) as img:
            width, height = img.size
            if width > TARGET_WIDTH:
                new_height = int(height * TARGET_WIDTH / width)
                img = img.resize((TARGET_WIDTH, new_height), Image.LANCZOS)
            img.save(dest_path, "WEBP", quality=WEBP_QUALITY)
        src_path.unlink()  # delete original
    except Exception as e:
        print(f"Error converting {src_path}: {e}")
        import shutil
        shutil.move(str(src_path), str(dest_path))

def process_all_pasted_images():
    if not images_dir.exists():
        images_dir.mkdir(parents=True)
        
    md_files = glob.glob(str(content_dir / "**" / "*.md"), recursive=True)
    img_pattern = r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]"
    
    total_processed = 0

    for file_path in md_files:
        md_path = Path(file_path)
        # Parse prefix: e.g. content/travel/251109 호수공원.md -> category=travel, name=251109 호수공원
        # We need relative path to content_dir
        rel_path = md_path.relative_to(content_dir)
        if len(rel_path.parts) < 2:
            continue # not in a category folder
        
        category = rel_path.parts[0]
        name_without_ext = md_path.stem
        
        match = re.match(r"^(\d{6})\s+(.+)$", name_without_ext)
        if match:
            date_str = match.group(1)
            name_str = match.group(2)
            safe_name = slugify_hangul(name_str)
            prefix = f"{category}_{date_str}_{safe_name}"
        else:
            safe_name = slugify_hangul(name_without_ext)
            prefix = f"{category}_{safe_name}"

        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        matches = list(re.finditer(img_pattern, content))
        if not matches:
            continue
            
        file_mapping = {}
        file_updated = False
        
        # Find current max index for this prefix
        existing_images = glob.glob(str(images_dir / f"{prefix}_*.webp"))
        max_index = 0
        for img_path in existing_images:
            basename = Path(img_path).stem
            # basename is like travel_251109_호수공원_1
            parts = basename.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                if idx > max_index:
                    max_index = idx

        for m in matches:
            old_filename = m.group(1).strip()
            old_path = blog_dir / old_filename
            
            if old_path.exists() and old_path.is_file():
                ext = old_path.suffix.lower()
                if ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"]:
                    if old_filename not in file_mapping:
                        max_index += 1
                        # Use .webp if Pillow is available, else keep original extension
                        new_ext = ".webp" if Image is not None else ext
                        new_filename = f"{prefix}_{max_index}{new_ext}"
                        new_path = images_dir / new_filename
                        
                        print(f"Processing/Moving {old_filename} -> {new_filename}")
                        # If image is already webp or PIL is missing, just move
                        if new_ext == ext and Image is None:
                            import shutil
                            shutil.move(str(old_path), str(new_path))
                        else:
                            convert_to_webp(old_path, new_path)
                        
                        # Just in case convert_to_webp failed and saved as .webp but it's really .png (handled by convert_to_webp now)
                        file_mapping[old_filename] = new_filename
                        total_processed += 1

        if not file_mapping:
            continue
            
        # Update the markdown content
        def replace_link(m):
            old_filename = m.group(1).strip()
            size_part = m.group(2)
            if old_filename in file_mapping:
                new_filename = file_mapping[old_filename]
                new_rel_path = f"../images/{new_filename}"
                if size_part:
                    return f"[[{new_rel_path}|{size_part}]]"
                else:
                    return f"[[{new_rel_path}]]"
            return m.group(0)
            
        new_content = re.sub(img_pattern, replace_link, content)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated {md_path.name}")

    if total_processed > 0:
        print(f"Total {total_processed} new pasted images processed.")

if __name__ == "__main__":
    process_all_pasted_images()
