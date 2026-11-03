class RunnerError(Exception):
    """Base class for errors the API maps to HTTP status codes."""


class ModelNotFound(RunnerError):
    """Unknown model id (404)."""


class ModelUnavailable(RunnerError):
    """Known model whose checkpoints are missing or failed to load (503)."""


class InvalidRequest(RunnerError):
    """Request is well-formed but not valid for this model (422)."""
