"""Integration tests for parameterized output directories (issue #323).

Verifies that step 1 honours an explicit ``run_name`` and that downstream
steps respect ``selected_runs`` so multiple parameter sets can coexist for the
same session without overwriting each other.
"""

import glob
import os
import shutil

import pytest
from conftest import STUBBED_TESTING_DATA

from guppy.testing.api import step1, step2, step3

CSV_SESSION = "csv/sample_data_csv_1"
CSV_STORENAMES = {
    "Sample_Control_Channel": "control_region",
    "Sample_Signal_Channel": "signal_region",
    "Sample_TTL": "ttl",
}


@pytest.fixture
def csv_session_copy(tmp_path):
    """Stage a clean copy of the CSV sample session and yield (base_dir, session_path)."""
    source = STUBBED_TESTING_DATA / CSV_SESSION
    base = tmp_path / "data_root"
    base.mkdir()
    destination = base / source.name
    shutil.copytree(source, destination)

    session_name = destination.name
    for stale in glob.glob(os.path.join(destination, f"{session_name}_output_*")):
        shutil.rmtree(stale)
    parameters = destination / "GuPPyParamtersUsed.json"
    if parameters.exists():
        parameters.unlink()

    return str(base), str(destination)


class TestStep1RunName:
    def test_explicit_run_name_creates_named_directory(self, csv_session_copy):
        base, session = csv_session_copy
        step1(
            base_dir=base,
            selected_folders=[session],
            store_id_to_store_label=CSV_STORENAMES,
            run_name="baseline",
        )
        expected = os.path.join(session, f"{os.path.basename(session)}_output_baseline")
        assert os.path.isdir(expected)
        assert os.path.exists(os.path.join(expected, "storesList.csv"))

    def test_two_run_names_coexist(self, csv_session_copy):
        base, session = csv_session_copy
        step1(
            base_dir=base,
            selected_folders=[session],
            store_id_to_store_label=CSV_STORENAMES,
            run_name="baseline",
        )
        step1(
            base_dir=base,
            selected_folders=[session],
            store_id_to_store_label=CSV_STORENAMES,
            run_name="strict",
        )
        session_basename = os.path.basename(session)
        assert os.path.isdir(os.path.join(session, f"{session_basename}_output_baseline"))
        assert os.path.isdir(os.path.join(session, f"{session_basename}_output_strict"))

    def test_create_policy_raises_on_existing_run_name(self, csv_session_copy):
        base, session = csv_session_copy
        step1(
            base_dir=base,
            selected_folders=[session],
            store_id_to_store_label=CSV_STORENAMES,
            run_name="baseline",
        )
        with pytest.raises(ValueError, match="already exists"):
            step1(
                base_dir=base,
                selected_folders=[session],
                store_id_to_store_label=CSV_STORENAMES,
                run_name="baseline",
                run_name_policy="create",
            )

    def test_overwrite_policy_replaces_existing_run_name(self, csv_session_copy):
        base, session = csv_session_copy
        step1(
            base_dir=base,
            selected_folders=[session],
            store_id_to_store_label=CSV_STORENAMES,
            run_name="baseline",
        )
        existing = os.path.join(session, f"{os.path.basename(session)}_output_baseline")
        marker = os.path.join(existing, "stale_marker.txt")
        with open(marker, "w") as marker_file:
            marker_file.write("stale")

        step1(
            base_dir=base,
            selected_folders=[session],
            store_id_to_store_label=CSV_STORENAMES,
            run_name="baseline",
            run_name_policy="overwrite",
        )

        assert os.path.isdir(existing)
        assert not os.path.exists(marker)

    def test_legacy_unspecified_run_name_uses_integer_suffix(self, csv_session_copy):
        base, session = csv_session_copy
        step1(base_dir=base, selected_folders=[session], store_id_to_store_label=CSV_STORENAMES)
        expected = os.path.join(session, f"{os.path.basename(session)}_output_1")
        assert os.path.isdir(expected)


class TestStep2SelectedRuns:
    def test_selected_runs_processes_only_chosen_dir(self, csv_session_copy):
        base, session = csv_session_copy
        for run_name in ("baseline", "strict"):
            step1(
                base_dir=base,
                selected_folders=[session],
                store_id_to_store_label=CSV_STORENAMES,
                run_name=run_name,
            )

        step2(
            base_dir=base,
            selected_folders=[session],
            selected_runs={session: ["baseline"]},
        )

        baseline_dir = os.path.join(session, f"{os.path.basename(session)}_output_baseline")
        strict_dir = os.path.join(session, f"{os.path.basename(session)}_output_strict")
        # Step 2 writes raw store HDF5 files alongside storesList.csv. The selected
        # baseline dir should have those files; the unselected strict dir should not.
        baseline_hdf5_files = glob.glob(os.path.join(baseline_dir, "*.hdf5"))
        strict_hdf5_files = glob.glob(os.path.join(strict_dir, "*.hdf5"))
        assert baseline_hdf5_files, "Step 2 produced no HDF5 outputs in the selected run"
        assert not strict_hdf5_files, "Step 2 wrote into the unselected run directory"

    def test_selected_runs_unknown_name_raises(self, csv_session_copy):
        base, session = csv_session_copy
        step1(
            base_dir=base,
            selected_folders=[session],
            store_id_to_store_label=CSV_STORENAMES,
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
                store_id_to_store_label=CSV_STORENAMES,
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

        baseline_dir = os.path.join(session, f"{os.path.basename(session)}_output_baseline")
        strict_dir = os.path.join(session, f"{os.path.basename(session)}_output_strict")
        baseline_zscore = glob.glob(os.path.join(baseline_dir, "z_score_*.hdf5"))
        strict_zscore = glob.glob(os.path.join(strict_dir, "z_score_*.hdf5"))
        assert baseline_zscore, "Step 3 produced no z-score outputs in the selected run"
        assert not strict_zscore, "Step 3 wrote z-score outputs to the unselected run directory"
