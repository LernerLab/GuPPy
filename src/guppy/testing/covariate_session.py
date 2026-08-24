"""Run the behavioral-covariate sample session through Steps 1-4.

The session is written by ``_write_covariate_csv_session`` in
``guppy/testing/scripts/create_stubbed_testing_data.py``. Both the covariate integration
tests and the documentation screenshot script need the same four steps run against it
with the same store labels and bin width, so that run lives here.
"""

import glob
import os
import shutil
from pathlib import Path

from guppy.testing.api import step1, step2, step3, step4
from guppy.utils.utils import parse_run_name

SESSION_NAME = "sample_data_csv_covariate_1"

RECORDING_SITE = "DMS"
COVARIATE_NAMES = ("akinesia", "grooming")
BIN_WIDTH = 50  # s — 600 s of recording gives 12 bins

STORE_ID_TO_STORE_LABEL = {
    "Sample_Control_Channel": f"control_{RECORDING_SITE}",
    "Sample_Signal_Channel": f"signal_{RECORDING_SITE}",
    "Sample_TTL": "ttl",
    "akinesia": "covariate_akinesia",
    "grooming": "covariate_grooming",
}


def locate_output_directory(*, session: str) -> str:
    """Return the run folder Step 1 created inside ``session``."""
    candidates = sorted(glob.glob(os.path.join(session, f"{os.path.basename(session)}_output_*")))
    assert candidates, f"no output directory was created in {session}"
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "storesList.csv")):
            return candidate
    raise AssertionError(f"no output directory in {session} contains storesList.csv")


def run_covariate_session(*, session_path: str | Path, base_directory: str | Path) -> str:
    """Copy the covariate sample session at ``session_path`` into ``base_directory``, run Steps 1-4, return its run folder.

    The steps require the session to sit directly under ``base_directory``.
    """
    session_path = Path(session_path)
    base_directory = Path(base_directory)
    session = base_directory / session_path.name
    shutil.copytree(session_path, session, ignore=shutil.ignore_patterns("*_output_*"))

    base_dir = str(base_directory)
    selected_folders = [str(session)]
    step1(
        base_dir=base_dir,
        selected_folders=selected_folders,
        store_id_to_store_label=STORE_ID_TO_STORE_LABEL,
    )

    output_directory = locate_output_directory(session=str(session))
    selected_runs = {str(session): [parse_run_name(output_directory)]}
    step2(base_dir=base_dir, selected_folders=selected_folders, selected_runs=selected_runs)
    step3(base_dir=base_dir, selected_folders=selected_folders, selected_runs=selected_runs)
    step4(
        base_dir=base_dir,
        selected_folders=selected_folders,
        selected_runs=selected_runs,
        compute_binned_metrics=True,
        binned_metrics_width=BIN_WIDTH,
    )
    return output_directory
