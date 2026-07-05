## agent/errors.py
class InvalidInput(Exception):
    """Raised by the input injector when a request fails validation."""

class HarnessTimeout(Exception):
    """Raised when a run exceeds the wall-clock budget (see harness._invoke_with_timeout)."""
