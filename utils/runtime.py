"""Runtime directory management for AgenticLoop.

All runtime data is stored under .aloop/ directory:
- config: Configuration file (created by config.py on first import)
- sessions/: YAML-based session persistence
- logs/: Log files (only created with --verbose)
- history: Interactive mode command history
"""

import os

RUNTIME_DIR = ".aloop"


def get_runtime_dir() -> str:
    """Get the runtime directory path.

    Returns:
        Path to .aloop directory
    """
    return RUNTIME_DIR


def get_config_file() -> str:
    """Get the configuration file path.

    Returns:
        Path to .aloop/config
    """
    return os.path.join(RUNTIME_DIR, "config")


def get_sessions_dir() -> str:
    """Get the sessions directory path.

    Returns:
        Path to .aloop/sessions/
    """
    return os.path.join(RUNTIME_DIR, "sessions")


def get_log_dir() -> str:
    """Get the log directory path.

    Returns:
        Path to .aloop/logs/
    """
    return os.path.join(RUNTIME_DIR, "logs")


def get_history_file() -> str:
    """Get the command history file path.

    Returns:
        Path to .aloop/history
    """
    return os.path.join(RUNTIME_DIR, "history")


def ensure_runtime_dirs(create_logs: bool = False) -> None:
    """Ensure runtime directories exist.

    Creates:
    - .aloop/sessions/
    - .aloop/logs/ (only if create_logs=True)

    Note: .aloop/config is created by config.py on first import.

    Args:
        create_logs: Whether to create the logs directory (for --verbose mode)
    """
    os.makedirs(os.path.join(RUNTIME_DIR, "sessions"), exist_ok=True)

    if create_logs:
        os.makedirs(os.path.join(RUNTIME_DIR, "logs"), exist_ok=True)
