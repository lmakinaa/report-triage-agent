"""Retries, consistency and injection safety."""
import time
from collections import Counter
from triage.classify import classify_case
from triage.errors import RateLimitError, TransientError
from triage.models import Case

def with_retries(fn, max_attempts: int=4, base_delay: float=1.0, sleep=time.sleep):
    """Call fn() and return its result, retrying only retryable errors.

- RateLimitError and TransientError: retry, waiting base_delay * 2**k before retry k+1
  (so 1.0, 2.0, 4.0 ... with the defaults). Call sleep(seconds) to wait.
- After max_attempts total calls, re-raise the last error.
- Any other exception is raised immediately, with no retry and no sleep."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except (RateLimitError, TransientError):
            if attempt == max_attempts - 1:
                raise
            sleep(base_delay * 2 ** attempt)

def flip_rate(case: Case, llm, runs: int=3, **kwargs) -> float:
    """Classify the same case `runs` times; return the share of runs whose label differs
from the most common label. Example: labels [A, A, B] -> 1/3. kwargs go to classify_case."""
    labels = [classify_case(case, llm, **kwargs).label for _ in range(runs)]
    most_common_count = Counter(labels).most_common(1)[0][1]
    return 1 - most_common_count / runs
