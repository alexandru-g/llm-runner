"""Background runner for local llama-server instances."""
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    # Single-sourced from pyproject.toml via the installed package metadata.
    __version__ = _pkg_version("llm-runner")
except PackageNotFoundError:  # running from a source checkout, not installed
    __version__ = "0.0.0+unknown"
