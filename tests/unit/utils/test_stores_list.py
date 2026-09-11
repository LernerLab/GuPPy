import numpy as np

from guppy.utils.stores_list import (
    COMBINED_STORES_LIST_FILENAME,
    read_stores_list,
    write_stores_list,
)


class TestWriteStoresList:
    def test_writes_two_rows_of_comma_separated_values(self, tmp_path):
        store_array = np.array([["Dv1A", "Dv2A", "PrtA"], ["control_DMS", "signal_DMS", "reward"]])

        write_stores_list(run_folder=str(tmp_path), store_array=store_array)

        assert (tmp_path / "storesList.csv").read_text() == "Dv1A,Dv2A,PrtA\ncontrol_DMS,signal_DMS,reward\n"

    def test_filename_selects_the_combined_sibling(self, tmp_path):
        store_array = np.array([["Dv1A"], ["control_DMS"]])

        write_stores_list(run_folder=str(tmp_path), store_array=store_array, filename=COMBINED_STORES_LIST_FILENAME)

        assert not (tmp_path / "storesList.csv").exists()
        assert (tmp_path / "combine_storesList.csv").read_text() == "Dv1A\ncontrol_DMS\n"


class TestReadStoresList:
    def test_reads_ids_and_labels(self, tmp_path):
        (tmp_path / "storesList.csv").write_text("Dv1A,Dv2A,PrtA\ncontrol_DMS,signal_DMS,reward\n")

        store_array = read_stores_list(run_folder=str(tmp_path))

        np.testing.assert_array_equal(
            store_array, np.array([["Dv1A", "Dv2A", "PrtA"], ["control_DMS", "signal_DMS", "reward"]])
        )

    def test_single_store_stays_two_dimensional(self, tmp_path):
        (tmp_path / "storesList.csv").write_text("Dv1A\ncontrol_DMS\n")

        store_array = read_stores_list(run_folder=str(tmp_path))

        assert store_array.shape == (2, 1)
        np.testing.assert_array_equal(store_array, np.array([["Dv1A"], ["control_DMS"]]))

    def test_round_trips_through_write(self, tmp_path):
        store_array = np.array([["Dv1A", "Dv2A"], ["control_DMS", "signal_DMS"]])

        write_stores_list(run_folder=str(tmp_path), store_array=store_array)

        np.testing.assert_array_equal(read_stores_list(run_folder=str(tmp_path)), store_array)
