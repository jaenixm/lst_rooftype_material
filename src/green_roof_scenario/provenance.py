"""Reproducibility metadata for scenario runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import pyproj
import rasterio

__all__ = ["environment_versions", "git_commit", "sha256_path"]


@lru_cache(maxsize=None)
def _sha256_file_cached(path_text: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    stat = path.stat()
    return _sha256_file_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=None)
def sha256_path(path_text: str) -> dict[str, object]:
    """Return a SHA-256 digest for a file or a deterministic directory tree."""

    path = Path(path_text).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return {"kind": "file", "sha256": _sha256_file(path), "size_bytes": path.stat().st_size}

    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    total_size = 0
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        item_digest = _sha256_file(item)
        size = item.stat().st_size
        total_size += size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(item_digest))
    return {
        "kind": "directory-tree",
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "size_bytes": total_size,
    }


def environment_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "geopandas", "rasterio", "rasterstats", "scikit-learn", "shapely"]
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "gdal": rasterio.__gdal_version__,
        "proj": pyproj.proj_version_str,
    }
    root = Path(__file__).resolve().parents[2]
    match = re.search(r'^version\s*=\s*"([^"]+)"', (root / "pyproject.toml").read_text(), re.MULTILINE)
    versions["green-roof-scenario-source"] = match.group(1) if match else "unknown"
    try:
        versions["green-roof-scenario-installed"] = importlib.metadata.version("green-roof-scenario")
    except importlib.metadata.PackageNotFoundError:
        versions["green-roof-scenario-installed"] = "not-installed"
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def git_commit() -> str | None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None
