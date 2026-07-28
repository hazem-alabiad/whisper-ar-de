"""Centralized logging configuration for whisper-tools.

Sets up a rotating file handler under ``logs/`` in the project root,
plus a concise console handler for WARNING-level messages.

Call ``setup_logging()`` once at application startup (in ``main()``).
All modules can then use the standard ``logging.getLogger(__name__)`` pattern.
"""

import logging
import logging.handlers
from datetime import UTC, datetime
from pathlib import Path

# ─── Public API ───────────────────────────────────────────────────────────────

LOG_DIR_NAME = "logs"


def setup_logging(
    log_dir: Path | None = None,
    console_level: int = logging.WARNING,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB per file
    backup_count: int = 5,                # keep last 5 rotated files
) -> Path:
    """Configure the root logger with a rotating file handler and a console handler.

    Args:
        log_dir: Directory to store log files. Defaults to ``<cwd>/logs``.
        console_level: Minimum level for console output (default WARNING).
        file_level: Minimum level written to the log file (default DEBUG).
        max_bytes: Maximum log file size before rotation.
        backup_count: Number of rotated backup files to retain.

    Returns:
        Path to the active log file.
    """
    if log_dir is None:
        # Resolve relative to cwd so the logs/ folder always sits next to
        # pyproject.toml, regardless of where Python is invoked from.
        log_dir = Path.cwd() / LOG_DIR_NAME

    log_dir.mkdir(parents=True, exist_ok=True)

    # Timestamped log file — one per session, rotated at max_bytes
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"whisper_tools_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)   # capture everything; handlers filter

    # ── File handler (DEBUG+) ──────────────────────────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(_detailed_formatter())

    # ── Console handler (WARNING+ by default) ──────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(_concise_formatter())

    # Remove any previously installed handlers (prevents duplicate lines on
    # re-entrant calls or interactive reloads)
    if root_logger.handlers:
        root_logger.handlers.clear()

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Silence overly chatty third-party loggers at DEBUG level
    for noisy_lib in ("transformers", "torch", "mlx", "httpx", "urllib3", "filelock"):
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)

    logging.getLogger(__name__).debug(
        "Logging initialised → %s (file) | console threshold: %s",
        log_file,
        logging.getLevelName(console_level),
    )

    return log_file


# ─── Formatter helpers ────────────────────────────────────────────────────────

def _detailed_formatter() -> logging.Formatter:
    """Rich formatter for log files — includes module path, line numbers."""
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _concise_formatter() -> logging.Formatter:
    """Minimal formatter for console — just level + message."""
    return logging.Formatter(
        fmt="[%(levelname)s] %(message)s",
    )
