"""Errors that map to an HTTP status and a stable machine-readable code."""
from __future__ import annotations


class ApiError(Exception):
    """Raised anywhere in request handling; the server turns it into
    ``{"error": {"code": ..., "message": ...}}`` with ``status``."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

    def to_json(self) -> dict:
        return {"error": {"code": self.code, "message": self.message}}


def invalid(message: str) -> ApiError:
    """A well-formed request the contract rejects (bad field, too many options, …)."""
    return ApiError(422, "invalid_request", message)


def too_long(message: str) -> ApiError:
    """The input does not fit the model's context; never silently truncated
    unless the model's own protocol truncates."""
    return ApiError(422, "input_too_long", message)


def backend_error(message: str) -> ApiError:
    return ApiError(502, "backend_error", message)
