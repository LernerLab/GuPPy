import csv
import os

import pytest

from guppy.testing.api import step2


@pytest.fixture(scope="function")
def storenames_map():
    return {
        "Dv1A": "control_DMS",
        "Dv2A": "signal_DMS",
        "PrtR": "RewardedPort",
    }


def test_step2_writes_storeslist(tmp_path, storenames_map):
    # Arrange: create base_dir with two session folders
    base_name = "data_root"
    base_dir = tmp_path / base_name
    base_dir.mkdir(parents=True, exist_ok=True)

    session_names = ["session1", "session2"]
    sessions = []
    for name in session_names:
        p = base_dir / name
        p.mkdir(parents=True, exist_ok=True)
        sessions.append(str(p))

    # Act: write storesList.csv headlessly
    step2(base_dir=str(base_dir), selected_folders=sessions, storenames_map=storenames_map)

    # Assert: each session has {session_basename}_output_1/storesList.csv with 2xN structure
    for s in sessions:
        basename = os.path.basename(s)
        out_dir = os.path.join(s, f"{basename}_output_1")
        out_fp = os.path.join(out_dir, "storesList.csv")

        assert os.path.isdir(out_dir), f"Missing output directory: {out_dir}"
        assert os.path.exists(out_fp), f"Missing storesList.csv: {out_fp}"

        with open(out_fp, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 2, f"Expected 2 rows (storenames, names_for_storenames), got {len(rows)}"
        assert rows[0] == list(storenames_map.keys()), "Row 0 (storenames) mismatch"
        assert rows[1] == list(storenames_map.values()), "Row 1 (names_for_storenames) mismatch"
