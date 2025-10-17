"""
Main entry point for GuPPy (Guided Photometry Analysis in Python)
"""

from . import logging_config

# Logging must be configured before importing application modules so that module-level loggers inherit the proper handlers and formatters
logging_config.setup_logging()

import panel as pn

from .savingInputParameters import savingInputParameters


def main():
    """Main entry point for GuPPy"""
    template = savingInputParameters()
    pn.serve(template, show=True)


if __name__ == "__main__":
    main()
