"""One /v1/systemone endpoint for calibrated decision models served by llama-server."""
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("systemone-proxy")
except PackageNotFoundError:  # running from a source tree
    __version__ = "0.0.0+unknown"
