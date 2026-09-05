import json
from pathlib import Path

import h5py
import pytest


@pytest.mark.parametrize(
    "step3_fixture_name, expected_recording_site, expected_ttl",
    [
        (
            "step3_output_csv",
            "region",
            "ttl",
        ),
        (
            "step3_output_tdt",
            "dms",
            "port_entries_dms",
        ),
        (
            "step3_output_npm",
            "region1",
            "ttl_region1",
        ),
        (
            "step3_output_doric",
            "region",
            "ttl",
        ),
        (
            "step3_output_nwb",
            "region",
            "ttl",
        ),
    ],
    ids=["csv_generic", "tdt_clean", "sample_npm_1", "sample_doric_1", "nwb_mock"],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_step3(step3_fixture_name, expected_recording_site, expected_ttl, request):
    """
    Validate Step 3 outputs for the representative integration sessions.
    """
    pipeline_state = request.getfixturevalue(step3_fixture_name)
    output_directory = str(pipeline_state["output_directory"])
    stores_file_path = Path(output_directory) / "storesList.csv"
    assert Path(stores_file_path).exists(), "Missing storesList.csv after Step 1/3/4"

    # Step 3 auto-writes the executed parameters into the output directory.
    parameters_file_path = Path(output_directory) / "GuPPyParamtersUsed.json"
    assert Path(parameters_file_path).exists(), "step 3 should write GuPPyParamtersUsed.json into the output directory"
    with Path(parameters_file_path).open() as parameters_file:
        saved_parameters = json.load(parameters_file)
    # The step-3 fixtures run with removeArtifacts disabled; the snapshot must reflect it.
    assert saved_parameters["removeArtifacts"] is False
    # The control fitting method defaults to IRWLS and is recorded in the snapshot.
    assert saved_parameters["control_fit_method"] == "IRWLS"

    # Ensure timeCorrection_<recording_site>.hdf5 exists with 'timestampNew'
    time_correction_file_path = Path(output_directory) / (f"timeCorrection_{expected_recording_site}.hdf5")
    assert Path(time_correction_file_path).exists(), f"Missing {time_correction_file_path}"
    with h5py.File(time_correction_file_path, "r") as time_correction_file:
        assert "timestampNew" in time_correction_file, f"Expected 'timestampNew' dataset in {time_correction_file_path}"

    # If TTLs exist, check their per-recording-site 'ts' outputs
    if expected_ttl is None:
        expected_ttl_names = []
    elif isinstance(expected_ttl, str):
        expected_ttl_names = [expected_ttl]
    else:
        expected_ttl_names = expected_ttl

    for expected_ttl_name in expected_ttl_names:
        ttl_file_path = Path(output_directory) / (f"{expected_ttl_name}_{expected_recording_site}.hdf5")
        assert Path(ttl_file_path).exists(), f"Missing TTL-aligned file {ttl_file_path}"
        with h5py.File(ttl_file_path, "r") as ttl_file:
            assert "ts" in ttl_file, f"Expected 'ts' dataset in {ttl_file_path}"
