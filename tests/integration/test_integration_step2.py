import csv
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from guppy.testing.api import step1, step2
from guppy.utils.progress import StepProgress, _current_step
from guppy_test_data import STUBBED_TESTING_DATA

from .integration_helpers import REPRESENTATIVE_SESSIONS, _locate_output_directory


@pytest.mark.parametrize(
    "step2_fixture_name",
    [
        "step2_output_csv",
        "step2_output_tdt",
        "step2_output_npm",
        "step2_output_doric",
        "step2_output_nwb",
    ],
    ids=["csv_generic", "tdt_clean", "sample_npm_1", "sample_doric_1", "nwb_mock"],
)
def test_step2(step2_fixture_name, request):
    """Validate Step 2 outputs for the representative integration sessions."""
    pipeline_state = request.getfixturevalue(step2_fixture_name)
    output_directory = str(pipeline_state["output_directory"])
    stores_file_path = Path(output_directory) / "storesList.csv"

    with Path(stores_file_path).open(newline="") as stores_file:
        stores_rows = list(csv.reader(stores_file))

    assert len(stores_rows) == 2, "storesList.csv should be 2 rows (store_ids, store_labels)"
    store_ids = stores_rows[0]
    assert store_ids, "Expected at least one store_id in storesList.csv"

    # Step 2 auto-writes the parameter snapshot into the selected output directory.
    assert (
        Path(output_directory) / "GuPPyParamtersUsed.json"
    ).exists(), "step 2 should write GuPPyParamtersUsed.json into the output directory"

    for store_id in store_ids:
        safe_store_id = store_id.replace("\\", "_").replace("/", "_")
        store_id_file_path = Path(output_directory) / (f"{safe_store_id}.hdf5")
        assert Path(store_id_file_path).exists(), f"Missing HDF5 for store_id {store_id!r} at {store_id_file_path}"

        with h5py.File(store_id_file_path, "r") as store_id_file:
            assert "timestamps" in store_id_file, "Expected 'timestamps' dataset in HDF5"


class TestStep2ProgressAccounting:
    """End-to-end verification that step 2 reports its progress against the total sample
    count across real extractors. TDT is sufficient — the accounting machinery itself is
    modality-agnostic and is covered per-extractor by ``count_samples`` unit tests; this
    test pins the wiring through ``orchestrate_read_raw_data`` against a real on-disk
    dataset.
    """

    def test_final_progress_value_matches_total_samples(self, tmp_path):
        step = StepProgress()
        token = _current_step.set(step)

        config = REPRESENTATIVE_SESSIONS["tdt"]
        source = Path(str(STUBBED_TESTING_DATA)) / config["session_subdir"]
        base_directory = tmp_path / "base"
        base_directory.mkdir()
        session_copy = base_directory / Path(source).name
        shutil.copytree(source, session_copy)

        step1(
            base_dir=str(base_directory),
            selected_folders=[str(session_copy)],
            store_id_to_store_label=config["store_id_to_store_label"],
        )
        output_directory = _locate_output_directory(session_copy=str(session_copy))

        from guppy.extractors.tdt_recording_extractor import TdtRecordingExtractor

        stores_list = np.genfromtxt(Path(output_directory) / "storesList.csv", dtype="str", delimiter=",").reshape(
            2, -1
        )
        extractor = TdtRecordingExtractor(str(session_copy))
        expected_total_samples = sum(extractor.count_samples(event=event) for event in np.unique(stores_list[0, :]))

        try:
            step2(
                base_dir=str(base_directory),
                selected_folders=[str(session_copy)],
                selected_runs={str(session_copy): [Path(output_directory).name.rsplit("_", 1)[-1]]},
            )
        finally:
            _current_step.reset(token)

        assert step.total == expected_total_samples
        assert step.value == expected_total_samples
