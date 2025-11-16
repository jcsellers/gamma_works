from __future__ import annotations

import base64
import fnmatch
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[import-not-found]

import textwrap


def _normalized_name(name: str) -> str:
    """Return the distribution-normalised project name."""

    return name.replace("-", "_")


def _iter_package_files(
    package_root: Path, data_patterns: Iterable[str]
) -> list[tuple[Path, str]]:
    """Return files that should be bundled into the wheel."""

    collected: list[tuple[Path, str]] = []
    patterns = list(data_patterns)
    for path in package_root.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(package_root)
        relative_name = relative.as_posix()
        if path.suffix == ".py" or any(
            fnmatch.fnmatch(relative_name, pattern) for pattern in patterns
        ):
            collected.append((path, f"xdte/{relative_name}"))
    return collected


def _build_metadata_text(project_table: dict[str, object]) -> str:
    """Construct a minimal core metadata payload."""

    name = project_table["name"]
    version = project_table["version"]
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {name}",
        f"Version: {version}",
    ]
    summary = project_table.get("description")
    if isinstance(summary, str):
        lines.append(f"Summary: {summary}")
    requires_python = project_table.get("requires-python")
    if isinstance(requires_python, str):
        lines.append(f"Requires-Python: {requires_python}")
    license_field = project_table.get("license")
    if isinstance(license_field, dict):
        text = license_field.get("text")
        if isinstance(text, str):
            lines.append(f"License: {text}")
    classifiers = project_table.get("classifiers", [])
    if isinstance(classifiers, list):
        for entry in classifiers:
            if isinstance(entry, str):
                lines.append(f"Classifier: {entry}")
    dependencies = project_table.get("dependencies", [])
    if isinstance(dependencies, list):
        for dependency in dependencies:
            if isinstance(dependency, str):
                lines.append(f"Requires-Dist: {dependency}")
    return "\n".join(lines) + "\n"


def _build_wheel(project_root: Path, wheel_dir: Path) -> Path:
    """Assemble a pure-Python wheel using the packaging metadata."""

    config = tomllib.loads(
        project_root.joinpath("pyproject.toml").read_text(encoding="utf-8")
    )
    project_table = config["project"]
    name = project_table["name"]
    version = project_table["version"]
    normalised = _normalized_name(name)
    package_data: Iterable[str] = []
    tool_table = config.get("tool", {})
    if isinstance(tool_table, dict):
        setuptools_table = tool_table.get("setuptools", {})
        if isinstance(setuptools_table, dict):
            package_data_table = setuptools_table.get("package-data", {})
            if isinstance(package_data_table, dict):
                patterns = package_data_table.get("xdte")
                if isinstance(patterns, list):
                    package_data = [
                        pattern for pattern in patterns if isinstance(pattern, str)
                    ]
    package_root = project_root / "src" / "xdte"
    files = _iter_package_files(package_root, package_data)

    dist_info = f"{normalised}-{version}.dist-info"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = wheel_dir / f"{normalised}-{version}-py3-none-any.whl"

    metadata_text = _build_metadata_text(project_table)
    wheel_metadata = (
        "Wheel-Version: 1.0\n"
        "Generator: xdte-tests\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n"
    )

    records: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, target in files:
            payload = source.read_bytes()
            archive.writestr(target, payload)
            records.append((target, payload))

        metadata_path = f"{dist_info}/METADATA"
        metadata_bytes = metadata_text.encode("utf-8")
        archive.writestr(metadata_path, metadata_bytes)
        records.append((metadata_path, metadata_bytes))

        wheel_path_inner = f"{dist_info}/WHEEL"
        wheel_bytes = wheel_metadata.encode("utf-8")
        archive.writestr(wheel_path_inner, wheel_bytes)
        records.append((wheel_path_inner, wheel_bytes))

        record_path = f"{dist_info}/RECORD"
        record_lines = []
        for target, payload in records:
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
                .decode("ascii")
                .rstrip("=")
            )
            record_lines.append(f"{target},sha256={digest},{len(payload)}")
        record_lines.append(f"{record_path},,")
        archive.writestr(record_path, "\n".join(record_lines) + "\n")

    return wheel_path


def _venv_python(venv_dir: Path) -> Path:
    """Return the path to the Python interpreter inside ``venv_dir``."""

    if os.name == "nt":  # pragma: no cover - Windows guard
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def test_packaged_json_available(tmp_path: Path) -> None:
    """Installing the wheel exposes the offline stub JSON payloads."""

    project_root = Path(__file__).resolve().parents[2]
    wheel_path = _build_wheel(project_root, tmp_path / "dist")

    venv_dir = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    python_exe = _venv_python(venv_dir)
    subprocess.run(
        [str(python_exe), "-m", "pip", "install", "--no-deps", str(wheel_path)],
        check=True,
    )

    verification = textwrap.dedent(
        """
        import json
        from importlib import resources

        daily = json.loads(
            resources.files('xdte')
            .joinpath('offline_daily_stub.json')
            .read_text(encoding='utf-8')
        )
        market = json.loads(
            resources.files('xdte')
            .joinpath('offline_market_stub.json')
            .read_text(encoding='utf-8')
        )
        assert daily['L1_TS'] == 0.9
        assert sorted(market) == ['11:00', '15:15']
        """
    )

    subprocess.run([str(python_exe), "-c", verification], check=True)
