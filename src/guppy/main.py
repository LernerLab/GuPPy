"""
Main entry point for GuPPy (Guided Photometry Analysis in Python)
"""

from . import logging_config

# Logging must be configured before importing application modules so that module-level loggers inherit the proper handlers and formatters
logging_config.setup_logging()

import argparse

import panel as pn

from .orchestration.home import build_homepage


def serve_app(*, start_path=None):
    """Serve the GuPPy application using Panel."""
    template = build_homepage(start_path=start_path)
    pn.serve(template, show=True)


def main():
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

    serve_app(start_path=args.start_path)


if __name__ == "__main__":
    main()
