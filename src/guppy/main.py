"""
Command-line entry point for GuPPy (Guided Photometry Analysis in Python).

Keep this module cheap to import. The pipeline steps run their multiprocessing pools under
the "spawn" start method, and a spawned worker is a fresh interpreter that must materialize
``__main__`` before it can unpickle its task. Here ``__main__`` is the ``guppy`` console
script, whose ``from guppy.main import main`` sits *above* its ``if __name__`` guard — so
every worker executes that import, on every pool creation. Importing Panel and the page
builders here therefore cost ~1.9s per pool instead of ~0.04s. The application itself lives
in ``guppy.app``, imported below only once the CLI has decided to serve it.
"""

import argparse

from . import logging_config

# At import scope, not inside main(), so that every process importing this module gets
# handlers -- including the spawned pool workers, whose logs would otherwise be discarded.
logging_config.setup_logging()


def main() -> None:
    """Main entry point for GuPPy.

    Supports command-line flags:
    - --export-logs: Export the log file to Desktop for sharing with support
    - --start-path: Set the initial directory for the folder selector
    - (no flags): Launch the GUI application
    """
    parser = argparse.ArgumentParser(description="GuPPy - Guided Photometry Analysis in Python")
    parser.add_argument(
        "--export-logs",
        action="store_true",
        help="Export log file to Desktop with timestamped name for support purposes",
    )
    parser.add_argument(
        "--start-path",
        type=str,
        default=None,
        help="Initial directory for the folder selector (defaults to home directory)",
    )

    args = parser.parse_args()

    if args.export_logs:
        logging_config.export_log_file()
        return

    # Deferred so that merely importing this module stays cheap -- see the module docstring.
    from .app import serve_app

    serve_app(start_path=args.start_path)


if __name__ == "__main__":
    main()
