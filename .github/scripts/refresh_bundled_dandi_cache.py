"""Refresh the DANDI photometry verdicts bundled with GuPPy.

Run by the refresh-bundled-dandi-cache workflow. Only dandisets that are new or have changed
since the bundle was last written are read, so a refresh after the first is short. A run that
reaches its time limit writes what it has settled, and the next run picks up from there.
"""

import argparse
import logging
import time

from guppy.utils.dandi_filter import refresh_bundled_verdicts


def main() -> None:
    """Refresh the bundle, stopping early when the time limit given on the command line runs out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--time-limit-minutes",
        type=float,
        default=None,
        help="Stop reading new dandisets after this many minutes and write what has been settled",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    deadline = time.monotonic() + args.time_limit_minutes * 60 if args.time_limit_minutes is not None else None
    verdicts = refresh_bundled_verdicts(
        on_verdict=lambda reference, holds: logging.info(
            "%s (%d assets): %s", reference.identifier, reference.asset_count, holds
        ),
        should_stop=lambda: deadline is not None and time.monotonic() > deadline,
    )
    logging.info(
        "Bundle holds %d dandiset verdicts, %d of them photometry",
        len(verdicts),
        sum(verdicts.values()),
    )


if __name__ == "__main__":
    main()
