import os

import h5py
import pytest


@pytest.mark.parametrize(
    "step4_fixture_name, expected_region, expected_ttl",
    [
        (
            "step4_output_csv",
            "region",
            "ttl",
        ),
        (
            "step4_output_tdt",
            "dms",
            "port_entries_dms",
        ),
        (
            "step4_output_npm",
            "region1",
            "ttl_region1",
        ),
        (
            "step4_output_doric",
            "region",
            "ttl",
        ),
    ],
    ids=["csv_generic", "tdt_clean", "sample_npm_1", "sample_doric_1"],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_step4(step4_fixture_name, expected_region, expected_ttl, request):
    """
    Validate Step 4 outputs for the representative integration sessions.
    """
    pipeline_state = request.getfixturevalue(step4_fixture_name)
    output_directory = str(pipeline_state["output_directory"])
    stores_file_path = os.path.join(output_directory, "storesList.csv")
    assert os.path.exists(stores_file_path), "Missing storesList.csv after Step 2/3/4"

    # Ensure timeCorrection_<region>.hdf5 exists with 'timestampNew'
    time_correction_file_path = os.path.join(output_directory, f"timeCorrection_{expected_region}.hdf5")
    assert os.path.exists(time_correction_file_path), f"Missing {time_correction_file_path}"
    with h5py.File(time_correction_file_path, "r") as time_correction_file:
        assert "timestampNew" in time_correction_file, f"Expected 'timestampNew' dataset in {time_correction_file_path}"

    # If TTLs exist, check their per-region 'ts' outputs
    if expected_ttl is None:
        expected_ttl_names = []
    elif isinstance(expected_ttl, str):
        expected_ttl_names = [expected_ttl]
    else:
        expected_ttl_names = expected_ttl

    for expected_ttl_name in expected_ttl_names:
        ttl_file_path = os.path.join(output_directory, f"{expected_ttl_name}_{expected_region}.hdf5")
        assert os.path.exists(ttl_file_path), f"Missing TTL-aligned file {ttl_file_path}"
        with h5py.File(ttl_file_path, "r") as ttl_file:
            assert "ts" in ttl_file, f"Expected 'ts' dataset in {ttl_file_path}"
