"""Sphinx configuration for the XDTE ML selector documentation."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "xdte"

sys.path.insert(0, str(SRC_ROOT))

project = "XDTE ML Selector"
author = "XDTE contributors"
copyright = f"{datetime.now():%Y}, {author}"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"


def run_apidoc(app: "Sphinx") -> None:
    """Generate API documentation with ``sphinx-apidoc`` prior to building."""

    from sphinx.ext.apidoc import main as apidoc_main

    output_dir = Path(__file__).parent / "api"
    module_dir = PACKAGE_ROOT
    apidoc_main(
        [
            "--force",
            "--module-first",
            "--no-toc",
            "-o",
            str(output_dir),
            str(module_dir),
        ]
    )


def setup(
    app: "Sphinx",
) -> None:  # pragma: no cover - exercised via documentation build
    """Register Sphinx event hooks."""

    app.connect("builder-inited", run_apidoc)
