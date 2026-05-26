#!/usr/bin/env python3
"""Sanitize generated outputs by redacting local absolute user paths.

Usage examples:
  python scripts/sanitize_generated_csv_outputs.py --check --root data/outputs
  python scripts/sanitize_generated_csv_outputs.py --write --root data/outputs
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

POSIX_USER_PATH = re.compile(r"/Users/(?!REDACTED/)[^/\r\n,\\]+/")
WINDOWS_USER_PATH = re.compile(
    r"[A-Za-z]:\\\\Users\\\\(?!REDACTED\\\\)[^\\\\\r\n,]+\\\\"
)


@dataclass
class FileResult:
    path: Path
    replacements: int


def sanitize_text(text: str) -> tuple[str, int]:
    """Redact local usernames embedded in absolute file paths."""
    out, n_posix = POSIX_USER_PATH.subn("/Users/REDACTED/", text)
    out, n_windows = WINDOWS_USER_PATH.subn("C:\\\\Users\\\\REDACTED\\\\", out)
    return out, n_posix + n_windows


def iter_sanitizable_files(root: Path) -> list[Path]:
    """Collect sanitizable text artifacts recursively under root in stable order."""
    out: list[Path] = []
    for suffix in ("*.csv", "*.json"):
        out.extend(p for p in root.rglob(suffix) if p.is_file())
    return sorted(set(out))


def run(root: Path, write: bool) -> tuple[list[FileResult], int]:
    """Scan CSV/JSON files and optionally persist sanitization changes."""
    changed_files: list[FileResult] = []
    total_replacements = 0

    for path in iter_sanitizable_files(root):
        original = path.read_text(encoding="utf-8", errors="ignore")
        sanitized, replacements = sanitize_text(original)
        if replacements == 0:
            continue

        total_replacements += replacements
        changed_files.append(FileResult(path=path, replacements=replacements))

        if write:
            path.write_text(sanitized, encoding="utf-8")

    return changed_files, total_replacements


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/outputs"),
        help="Root directory to scan recursively for CSV/JSON files.",
    )
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument(
        "--check",
        action="store_true",
        help="Check mode: report files that would be sanitized and exit non-zero if any are found.",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="Write mode: apply sanitization in place.",
    )
    return parser


def main() -> int:
    """Entrypoint for CLI execution."""
    parser = build_parser()
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        parser.error(f"Root directory does not exist: {root}")

    write = bool(args.write)
    changed_files, total_replacements = run(root=root, write=write)

    if not changed_files:
        print("No sanitization needed.")
        return 0

    action = "Sanitized" if write else "Would sanitize"
    print(f"{action} {len(changed_files)} file(s) with {total_replacements} replacement(s):")
    for result in changed_files:
        print(f"- {result.path} ({result.replacements} replacement(s))")

    if write:
        return 0

    # Check mode fails if there is anything to sanitize.
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
