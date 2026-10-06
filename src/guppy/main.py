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
from importlib.metadata import version
from pathlib import Path

from . import logging_config

# At import scope, not inside main(), so that every process importing this module gets
# handlers -- including the spawned pool workers, whose logs would otherwise be discarded.
logging_config.setup_logging()


def _resolve_root_folder(*, parser: argparse.ArgumentParser, flag: str, value: str | None) -> str | None:
    """Return a root folder named on the command line as an absolute path, refusing one that does not exist.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser that read ``value``, used to report a missing folder as a usage error.
    flag : str
        The flag ``value`` was given to, named in the error.
    value : str or None
        The folder as typed, possibly relative or starting with ``~``; ``None`` when the flag was not given.

    Returns
    -------
    str or None
        The absolute folder, or ``None`` when the flag was not given.
    """
    if value is None:
        return None
    folder = Path(value).expanduser().resolve()
    if not folder.is_dir():
        parser.error(f"{flag} '{value}' is not an existing folder. Create it first, or name a folder that exists.")
    return str(folder)


def main(*, argv: list[str] | None = None) -> None:
    """Main entry point for GuPPy.

    Supports command-line flags:
    - --version: Print the installed GuPPy version and exit
    - --export-logs: Export the log file to Desktop for sharing with support
    - --clear-dandi-cache: Delete the cached DANDI photometry scan results and exit
    - --input-root: Set the folder the session folders live under
    - --output-root: Set the folder the mirrored output tree is written into
    - (no flags): Launch the GUI application

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector to parse. When None (the console-script case) argparse reads
        ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description="GuPPy - Guided Photometry Analysis in Python")
    parser.add_argument(
        "--version",
        action="version",
        version=f"GuPPy {version('guppy-neuro')}",
        help="Print the installed GuPPy version and exit",
    )
    parser.add_argument(
        "--export-logs",
        action="store_true",
        help="Export log file to Desktop with timestamped name for support purposes",
    )
    parser.add_argument(
        "--clear-dandi-cache",
        action="store_true",
        help="Delete the cached results of DANDI fiber photometry scans, so the next scan reads every file again",
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default=None,
        help="Folder your session folders live under; remembered for the next launch",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Folder the mirrored output tree is written into; remembered for the next launch",
    )

    args = parser.parse_args(argv)

    if args.export_logs:
        logging_config.export_log_file()
        return

    if args.clear_dandi_cache:
        # Deferred for the same reason as the app import below: the scan module pulls in h5py.
        from .utils import dandi_filter

        cleared = dandi_filter.clear_verdict_cache()
        if cleared is None:
            print(f"No DANDI scan cache to clear at {dandi_filter.default_verdict_cache_path()}")
        else:
            print(f"Cleared the DANDI scan cache at {cleared}")
        return

    # Deferred so that merely importing this module stays cheap -- see the module docstring.
    from .app import serve_app

    serve_app(
        input_root_folder=_resolve_root_folder(parser=parser, flag="--input-root", value=args.input_root),
        output_root_folder=_resolve_root_folder(parser=parser, flag="--output-root", value=args.output_root),
    )


if __name__ == "__main__":
    main()
