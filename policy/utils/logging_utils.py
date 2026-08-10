from __future__ import annotations

import logging

import rich.logging

import policy


def setup_logging(log_level: str, global_log_level: str = "WARNING") -> None:
    """Installs a `rich` handler and sets the project's log level."""
    logging.basicConfig(
        level=global_log_level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        force=True,
        handlers=[
            rich.logging.RichHandler(
                markup=True,
                rich_tracebacks=True,
                tracebacks_width=100,
                tracebacks_show_locals=False,
            )
        ],
    )

    logging.getLogger(policy.__name__).setLevel(log_level.upper())
