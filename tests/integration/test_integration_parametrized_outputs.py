"""Integration tests for parameterized output directories (issue #323).

Verifies that step 1 honours an explicit ``run_name`` and that downstream
steps respect ``selected_runs`` so multiple parameter sets can coexist for the
same session without overwriting each other.
"""

import shutil
from pathlib import Path

import pytest

from guppy.testing import default_output_root_folder
from guppy.testing.api import step1, step2, step3
from guppy.utils.utils import run_folder_for_run
from guppy_test_data import STUBBED_TESTING_DATA

CSV_SESSION = "csv/sample_data_csv_1"
CSV_STORE_ID_TO_STORE_LABEL = {
    "Sample_Control_Channel": "control_region",
    "Sample_Signal_Channel": "signal_region",
    "Sample_TTL": "ttl",
}


@pytest.fixture
def csv_session_copy(tmp_path):
    """Stage a clean copy of the CSV sample session and yield (base_dir, session_path)."""
    source = STUBBED_TESTING_DATA / CSV_SESSION
    base = tmp_path / "input_root_folder"
    base.mkdir()
    destination = base / source.name
    shutil.copytree(source, destination)

    session_name = destination.name
    for stale in list(Path(destination).glob(f"{session_name}_output_*")):
        shutil.rmtree(stale)
    parameters = destination / "GuPPyParamtersUsed.json"
    if parameters.exists():
        parameters.unlink()

    return str(base), str(destination)


def _run_folder(session, run_name):
    """The run folder Step 1 creates for ``session`` under the headless steps' output layout."""
    input_root_folder = str(Path(session).parent)
    return Path(
        run_folder_for_run(
            session,
            run_name,
            output_root_folder=default_output_root_folder(base_dir=input_root_folder),
            input_root_folder=input_root_folder,
        )
    )


class TestStep1RunName:
    def test_explicit_run_name_creates_named_directory(self, csv_session_copy):
        base, session = csv_session_copy
        step1(
            base_dir=base,
            selected_folders=[session],
            store_id_to_store_label=CSV_STORE_ID_TO_STORE_LABEL,
            run_name="baseline",
        )
        expected = _run_folder(session, "baseline")
        assert Path(expected).is_dir()
        assert (Path(expected) / "storesList.csv").exists()

    def test_two_run_names_coexist(self, csv_session_copy):
        base, session = csv_session_copy
        step1(
            base_dir=base,
            selected_folders=[session],
            store_id_to_store_label=CSV_STORE_ID_TO_STORE_LABEL,
            run_name="baseline",
        )
        step1(
            base_dir=base,
            selected_folders=[session],
            store_id_to_store_label=CSV_STORE_ID_TO_STORE_LABEL,
            run_name="strict",
        )
        assert (_run_folder(session, "baseline")).is_dir()
        assert (_run_folder(session, "strict")).is_dir()

    def test_create_policy_raises_on_existing_run_name(self, csv_session_copy):
        base, session = csv_session_copy
        step1(
            base_dir=base,
            selected_folders=[session],
            store_id_to_store_label=CSV_STORE_ID_TO_STORE_LABEL,
            run_name="baseline",
        )
        with pytest.raises(ValueError, match="already exists"):
            step1(
                base_dir=base,
                selected_folders=[session],
                store_id_to_store_label=CSV_STORE_ID_TO_STORE_LABEL,
                run_name="baseline",
                run_name_policy="create",
            )

    def test_overwrite_policy_replaces_existing_run_name(self, csv_session_copy):
        base, session = csv_session_copy
        step1(
            base_dir=base,
            selected_folders=[session],
            store_id_to_store_label=CSV_STORE_ID_TO_STORE_LABEL,
            run_name="baseline",
        )
        existing = _run_folder(session, "baseline")
        marker = Path(existing) / "stale_marker.txt"
        with Path(marker).open("w") as marker_file:
            marker_file.write("stale")

        step1(
            base_dir=base,
            selected_folders=[session],
            store_id_to_store_label=CSV_STORE_ID_TO_STORE_LABEL,
            run_name="baseline",
            run_name_policy="overwrite",
        )

        assert Path(existing).is_dir()
        assert not Path(marker).exists()

    def test_legacy_unspecified_run_name_uses_integer_suffix(self, csv_session_copy):
        base, session = csv_session_copy
        step1(base_dir=base, selected_folders=[session], store_id_to_store_label=CSV_STORE_ID_TO_STORE_LABEL)
        expected = _run_folder(session, "1")
        assert Path(expected).is_dir()


class TestStep2SelectedRuns:
    def test_selected_runs_processes_only_chosen_dir(self, csv_session_copy):
        base, session = csv_session_copy
        for run_name in ("baseline", "strict"):
            step1(
                base_dir=base,
                selected_folders=[session],
                store_id_to_store_label=CSV_STORE_ID_TO_STORE_LABEL,
                run_name=run_name,
            )

        step2(
            base_dir=base,
            selected_folders=[session],
            selected_runs={session: ["baseline"]},
        )

        baseline_dir = _run_folder(session, "baseline")
        strict_dir = _run_folder(session, "strict")
        # Step 2 writes raw store HDF5 files alongside storesList.csv. The selected
        # baseline dir should have those files; the unselected strict dir should not.
        baseline_hdf5_files = list(Path(baseline_dir).glob("*.hdf5"))
        strict_hdf5_files = list(Path(strict_dir).glob("*.hdf5"))
        assert baseline_hdf5_files, "Step 2 produced no HDF5 outputs in the selected run"
        assert not strict_hdf5_files, "Step 2 wrote into the unselected run directory"

    def test_selected_runs_unknown_name_raises(self, csv_session_copy):
        base, session = csv_session_copy
        step1(
            base_dir=base,
            selected_folders=[session],
            store_id_to_store_label=CSV_STORE_ID_TO_STORE_LABEL,
            run_name="baseline",
        )
        with pytest.raises(ValueError, match="Output directory not found"):
            step2(
                base_dir=base,
                selected_folders=[session],
                selected_runs={session: ["nonexistent"]},
            )


class TestStep3SelectedRuns:
    def test_selected_runs_processes_only_chosen_dir(self, csv_session_copy):
        base, session = csv_session_copy
        for run_name in ("baseline", "strict"):
            step1(
                base_dir=base,
                selected_folders=[session],
                store_id_to_store_label=CSV_STORE_ID_TO_STORE_LABEL,
                run_name=run_name,
            )
        # Run step 2 only for the dir that step 3 will operate on; the unselected
        # strict dir is left without raw HDF5 files so we can verify step 3 ignores it.
        step2(
            base_dir=base,
            selected_folders=[session],
            selected_runs={session: ["baseline"]},
        )
        step3(
            base_dir=base,
            selected_folders=[session],
            selected_runs={session: ["baseline"]},
        )

        baseline_dir = _run_folder(session, "baseline")
        strict_dir = _run_folder(session, "strict")
        baseline_zscore = list(Path(baseline_dir).glob("z_score_*.hdf5"))
        strict_zscore = list(Path(strict_dir).glob("z_score_*.hdf5"))
        assert baseline_zscore, "Step 3 produced no z-score outputs in the selected run"
        assert not strict_zscore, "Step 3 wrote z-score outputs to the unselected run directory"
