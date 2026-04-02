import csv
import os

import h5py
import pytest


@pytest.mark.parametrize(
    "step3_fixture_name",
    [
        "step3_output_csv",
        "step3_output_tdt",
        "step3_output_npm",
        "step3_output_doric",
    ],
    ids=["csv_generic", "tdt_clean", "sample_npm_1", "sample_doric_1"],
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
