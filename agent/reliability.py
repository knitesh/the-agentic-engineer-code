## agent/reliability.py — provider-aware error classification.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

def is_retryable(err: Exception) -> bool:
    status = getattr(err, "status_code", None) or getattr(err, "http_status", None)
    if status in RETRYABLE_STATUS:
        return True
    # Timeouts and connection resets are transient by nature.
    if isinstance(err, (TimeoutError, ConnectionError)):
        return True
    # Default: do NOT retry. Unknown errors are treated as fatal so we fail
    # fast and loud rather than burning the budget on a doomed call.
    return False


import random

def backoff_delay(attempt: int, base: float, retry_after: float | None) -> float:
    if retry_after is not None:
        return retry_after                       # provider told us; obey it
    # Exponential backoff with full jitter (spreads the thundering herd).
    return random.uniform(0, base * (2 ** attempt))


import threading, time

class RateLimiter:
    """Token-bucket limiter: caps sustained request rate, allows small bursts.
    Wrap each provider call's acquire() so the whole process stays under the
    provider's requests-per-minute ceiling regardless of worker count."""
    def __init__(self, rate_per_sec: float, burst: int):
        self.rate, self.capacity = rate_per_sec, burst
        self.tokens = float(burst)
        self.updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        # Compute any wait UNDER the lock, then sleep OUTSIDE it. Sleeping while
        # holding the lock would serialize every caller and defeat concurrency.
        while True:
            with self._lock:
                now = time.monotonic()
                self.tokens = min(self.capacity,
                                  self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                sleep_for = (1 - self.tokens) / self.rate
            time.sleep(sleep_for)        # lock released; other threads proceed


## Illustrative (11.2): a write tool made retry-safe with an idempotency key.
## Requires your email provider's client — shown for the pattern, not wired in.
import hashlib

def send_email(*, to: str, body: str, state: dict) -> str:
    idem_key = hashlib.sha256(
        f"{state['thread_id']}:{to}:{body}".encode()
    ).hexdigest()
    resp = email_api.send(to=to, body=body, idempotency_key=idem_key)
    return f"sent ({resp.id})"
