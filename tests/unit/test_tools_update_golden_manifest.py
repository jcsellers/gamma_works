"""Tests for the golden manifest refresh helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import update_golden_manifest as ugm


def test_discover_books_prefers_directories(tmp_path: Path) -> None:
    source = tmp_path / "books"
    source.mkdir()
    (source / "CALLS_0DTE_11").mkdir()
    (source / "PUTS_0DTE_11").mkdir()
    (source / "manifest.json").write_text("{}", encoding="utf-8")

    books = ugm._discover_books_in_directory(source)

    assert books == ["CALLS_0DTE_11", "PUTS_0DTE_11"]


def test_discover_books_falls_back_to_files(tmp_path: Path) -> None:
    source = tmp_path / "files"
    source.mkdir()
    (source / "CALLS_0DTE_11.json").write_text("{}", encoding="utf-8")
    (source / "PUTS_0DTE_11.csv").write_text("", encoding="utf-8")

    books = ugm._discover_books_in_directory(source)

    assert books == ["CALLS_0DTE_11", "PUTS_0DTE_11"]


@pytest.mark.parametrize(
    "sources,expected",
    [
        ((("CALLS",), ("PUTS",)), ["CALLS", "PUTS"]),
        ((("CALLS", "CALLS"), ("PUTS", "CALLS"), ()), ["CALLS", "PUTS"]),
        ((("  CALLS  ",), ("", "PUTS")), ["CALLS", "PUTS"]),
    ],
)
def test_merge_book_sources(
    sources: tuple[tuple[str, ...], ...], expected: list[str]
) -> None:
    assert ugm._merge_book_sources(*sources) == expected
