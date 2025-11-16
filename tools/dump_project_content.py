"""Utility script to dump project source code and tests into flat text files.

This module collects all Python source files within the ``src`` and ``tests``
directories and writes their concatenated contents to ``source_dump.txt`` and
``tests_dump.txt`` respectively. Each file's contents are prefixed with a header
indicating the relative path to aid navigation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
TESTS_DIR = REPO_ROOT / "tests"
SOURCE_DUMP_PATH = REPO_ROOT / "source_dump.txt"
TESTS_DUMP_PATH = REPO_ROOT / "tests_dump.txt"


def iter_python_files(directory: Path) -> List[Path]:
    """Return a list of Python files contained in *directory* sorted by relative path."""

    if not directory.exists():
        return []

    files: List[Path] = [
        path
        for path in directory.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    files.sort(key=lambda path: path.relative_to(REPO_ROOT).as_posix())
    return files


def dump_files(file_paths: Iterable[Path], destination: Path) -> None:
    """Write *file_paths* contents into *destination* with informative headers."""

    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as outfile:
        for path in file_paths:
            relative = path.relative_to(REPO_ROOT).as_posix()
            outfile.write(f"# File: {relative}\n\n")
            text = path.read_text(encoding="utf-8")
            outfile.write(text)
            if not text.endswith("\n"):
                outfile.write("\n")
            outfile.write("\n")


def main() -> None:
    source_files = iter_python_files(SRC_DIR)
    test_files = iter_python_files(TESTS_DIR)

    dump_files(source_files, SOURCE_DUMP_PATH)
    dump_files(test_files, TESTS_DUMP_PATH)


if __name__ == "__main__":
    main()
