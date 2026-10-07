"""Refresh the DANDI photometry verdicts bundled with GuPPy.

Run by the refresh-bundled-dandi-cache workflow, and by hand for a refresh too long for it. Only
dandisets that are new or have changed since the bundle was last written are read, so a refresh
after the first is short. The bundle is rewritten as each dandiset settles, so a run that is
interrupted or reaches its time limit keeps what it settled and the next run picks up from there.
"""

import argparse
import logging
import time

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from guppy.utils.dandi_filter import (
    DandisetReference,
    list_archive_dandisets,
    refresh_bundled_verdicts,
)


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
    references = list_archive_dandisets()
    logging.info("Listed %d dandisets", len(references))

    # disable=None turns the bar off when the output is not a terminal, as in the workflow's log.
    with tqdm(total=len(references), unit="dandiset", disable=None) as progress_bar, logging_redirect_tqdm():

        def on_verdict(reference: DandisetReference, holds: bool | None) -> None:
            logging.info("%s (%d assets): %s", reference.identifier, reference.asset_count, holds)
            progress_bar.update()

        verdicts = refresh_bundled_verdicts(
            references,
            on_verdict=on_verdict,
            should_stop=lambda: deadline is not None and time.monotonic() > deadline,
        )
    logging.info(
        "Bundle holds %d dandiset verdicts, %d of them photometry",
        len(verdicts),
        sum(verdicts.values()),
    )


if __name__ == "__main__":
    main()
