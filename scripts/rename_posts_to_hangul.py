#!/usr/bin/env python3
"""Rename percent-encoded Markdown filenames in blog/content/posts to Hangul.

The script decodes URL-encoded characters (e.g. %ec%95%88%eb%85%95) in
Markdown filenames so that they use the original Hangul titles.
Run with --dry-run first to review the planned changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import unquote


def find_candidates(base: Path, recursive: bool) -> list[tuple[Path, Path]]:
    """Return rename candidates as (source, destination) tuples."""
    pattern = "**/*.md" if recursive else "*.md"
    candidates: list[tuple[Path, Path]] = []

    for src in sorted(base.glob(pattern)):
        decoded_name = unquote(src.name)

        # Skip files that do not need decoding.
        if decoded_name == src.name:
            continue

        dest = src.with_name(decoded_name)
        candidates.append((src, dest))

    return candidates


def filter_conflicts(candidates: list[tuple[Path, Path]]) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Remove conflicting candidates and return (clean_candidates, warnings)."""
    warnings: list[str] = []
    clean: list[tuple[Path, Path]] = []
    dest_seen: dict[Path, Path] = {}

    for src, dest in candidates:
        if dest.exists() and dest != src:
            warnings.append(f"skip (already exists): {src} -> {dest}")
            continue

        other = dest_seen.get(dest)
        if other is not None and other != src:
            warnings.append(f"skip (duplicate target): {src} -> {dest}")
            continue

        dest_seen[dest] = src
        clean.append((src, dest))

    return clean, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default="blog/content/posts",
        type=Path,
        help="Directory that contains the Markdown posts",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process nested directories under the target",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the renames without applying them",
    )
    args = parser.parse_args()

    base = args.target.resolve()
    if not base.is_dir():
        print(f"Target directory not found: {base}", file=sys.stderr)
        return 1

    candidates = find_candidates(base, args.recursive)
    if not candidates:
        print("No percent-encoded Markdown filenames found.")
        return 0

    clean_candidates, warnings = filter_conflicts(candidates)

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  {warning}")
        print()

    if not clean_candidates:
        print("No safe renames to apply.")
        return 1

    print("Planned renames:")
    for src, dest in clean_candidates:
        rel_src = src.relative_to(base)
        rel_dest = dest.relative_to(base)
        print(f"  {rel_src} -> {rel_dest}")

    if args.dry_run:
        print("\nDry-run complete. Re-run without --dry-run to apply changes.")
        return 0

    for src, dest in clean_candidates:
        src.rename(dest)

    print("\nRenaming complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
