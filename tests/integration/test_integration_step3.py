import csv
import os
import shutil

import h5py
import numpy as np
import pytest
from conftest import (
    REPRESENTATIVE_SESSIONS,
    STUBBED_TESTING_DATA,
    _locate_output_directory,
)

from guppy.testing.api import step2, step3


@pytest.mark.parametrize(
    "step3_fixture_name",
    [
        "step3_output_csv",
        "step3_output_tdt",
        "step3_output_npm",
        "step3_output_doric",
        "step3_output_nwb",
    ],
    ids=["csv_generic", "tdt_clean", "sample_npm_1", "sample_doric_1", "nwb_mock"],
)
def test_step3(step3_fixture_name, request):
    """Validate Step 3 outputs for the representative integration sessions."""
    pipeline_state = request.getfixturevalue(step3_fixture_name)
    output_directory = str(pipeline_state["output_directory"])
    stores_file_path = os.path.join(output_directory, "storesList.csv")

    with open(stores_file_path, newline="") as stores_file:
        stores_rows = list(csv.reader(stores_file))

    assert len(stores_rows) == 2, "storesList.csv should be 2 rows (storenames, names_for_storenames)"
    storenames = stores_rows[0]
    assert storenames, "Expected at least one storename in storesList.csv"

    for storename in storenames:
        safe_storename = storename.replace("\\", "_").replace("/", "_")
        storename_file_path = os.path.join(output_directory, f"{safe_storename}.hdf5")
        assert os.path.exists(storename_file_path), f"Missing HDF5 for storename {storename!r} at {storename_file_path}"

        with h5py.File(storename_file_path, "r") as storename_file:
            assert "timestamps" in storename_file, "Expected 'timestamps' dataset in HDF5"


class TestStep3ProgressFileAccounting:
    """End-to-end verification that step 3 reconciles its progress file to the
    total sample count across real extractors. TDT is sufficient — the
    accounting machinery itself is modality-agnostic and is covered per-extractor
    by ``count_samples`` unit tests; this test pins the wiring through
    ``orchestrate_read_raw_data`` against a real on-disk dataset.
    """

    def test_final_progress_value_matches_total_samples(self, tmp_path, monkeypatch):
        from guppy.frontend import progress as progress_module
        from guppy.orchestration import read_raw_data as read_raw_data_module

        progress_file = tmp_path / "pb_steps.txt"
        monkeypatch.setattr(read_raw_data_module, "PB_STEPS_FILE", str(progress_file))
        monkeypatch.setattr(progress_module, "PB_STEPS_FILE", str(progress_file))

        config = REPRESENTATIVE_SESSIONS["tdt"]
        source = os.path.join(str(STUBBED_TESTING_DATA), config["session_subdir"])
        base_directory = tmp_path / "base"
        base_directory.mkdir()
        session_copy = base_directory / os.path.basename(source)
        shutil.copytree(source, session_copy)

        step2(
            base_dir=str(base_directory),
            selected_folders=[str(session_copy)],
            storenames_map=config["storenames_map"],
        )
        output_directory = _locate_output_directory(session_copy=str(session_copy))

        from guppy.extractors.tdt_recording_extractor import TdtRecordingExtractor

        stores_list = np.genfromtxt(
            os.path.join(output_directory, "storesList.csv"), dtype="str", delimiter=","
        ).reshape(2, -1)
        extractor = TdtRecordingExtractor(str(session_copy))
        expected_total_samples = sum(extractor.count_samples(event=event) for event in np.unique(stores_list[0, :]))

        step3(
            base_dir=str(base_directory),
            selected_folders=[str(session_copy)],
            selected_runs={str(session_copy): [os.path.basename(output_directory).rsplit("_", 1)[-1]]},
        )

        written_lines = [line.strip() for line in progress_file.read_text().splitlines() if line.strip()]
        written_values = [int(value) for value in written_lines]
        assert written_values[0] == expected_total_samples * 10
        assert written_values[-1] == expected_total_samples * 10
