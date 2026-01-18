"""Retry utilities for LLM API calls with exponential backoff."""

import time
from functools import wraps
from typing import Callable, TypeVar

from config import Config
from utils import get_logger

logger = get_logger(__name__)
T = TypeVar("T")


def is_rate_limit_error(error: Exception) -> bool:
    """Check if an error is a rate limit error.

    Args:
        error: Exception to check

    Returns:
        True if this is a rate limit error
    """
    error_str = str(error).lower()

    # Common rate limit indicators
    rate_limit_indicators = [
        "429",
        "rate limit",
        "quota",
        "too many requests",
        "resourceexhausted",
    ]

    return any(indicator in error_str for indicator in rate_limit_indicators)


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable.

    Args:
        error: Exception to check

    Returns:
        True if this error should trigger a retry
    """
    # Rate limit errors are always retryable
    if is_rate_limit_error(error):
        return True

    error_str = str(error).lower()
    error_type = type(error).__name__

    # LiteLLM-specific errors
    if "RateLimitError" in error_type or "APIConnectionError" in error_type:
        return True

    # Other retryable errors
    retryable_indicators = [
        "timeout",
        "connection",
        "server error",
        "500",
        "502",
        "503",
        "504",
    ]

    return any(indicator in error_str for indicator in retryable_indicators)


def with_retry():
    """Decorator to add retry logic with exponential backoff.

    Uses Config.RETRY_* settings for retry configuration.

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error = None
            max_retries = Config.RETRY_MAX_ATTEMPTS

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    # Don't retry on last attempt
                    if attempt == max_retries:
                        break

                    # Only retry if error is retryable
                    if not is_retryable_error(e):
                        raise

                    # Calculate delay using Config
                    delay = Config.get_retry_delay(attempt)

                    # Log retry attempt
                    error_type = "Rate limit" if is_rate_limit_error(e) else "Retryable"
                    logger.warning(f"{error_type} error: {str(e)}")
                    logger.warning(
                        f"Retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})"
                    )

                    # Wait before retry
                    time.sleep(delay)

            # All retries exhausted
            raise last_error

        return wrapper

    return decorator


def retry_with_backoff(func: Callable[..., T], *args, **kwargs) -> T:
    """Execute a function with retry logic.

    Args:
        func: Function to execute
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Result from func

    Raises:
        Last exception if all retries fail
    """
    decorated_func = with_retry()(func)
    return decorated_func(*args, **kwargs)
