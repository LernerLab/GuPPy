import numpy as np
import panel as pn
import pytest

from guppy.orchestration.raw_data_view import open_raw_data_view
from guppy.utils import page_links
from guppy.utils._hdf5_io import write_hdf5
from guppy.utils.stores_list import write_stores_list


@pytest.fixture
def session_with_two_runs(tmp_path):
    session = tmp_path / "session_a"
    for run_name in ("1", "2"):
        run_folder = session / f"output_{run_name}"
        run_folder.mkdir(parents=True)
        write_stores_list(run_folder=run_folder, store_array=np.array([["sig"], ["signal_A"]]))
        write_hdf5(np.array([0.0, 1.0]), "sig", str(run_folder), "timestamps")
        write_hdf5(np.array([1.0, 2.0]), "sig", str(run_folder), "data")
    return session


def test_offers_every_selected_run_even_when_combining(panel_extension, base_input_parameters, session_with_two_runs):
    input_parameters = {
        **base_input_parameters,
        "combine_data": True,
        "selected_runs": {str(session_with_two_runs): ["1", "2"]},
    }

    with page_links.collect() as pages:
        open_raw_data_view([str(session_with_two_runs)], input_parameters)

    (page,) = pages
    assert page.name == "Raw Data"
    (run_folder_select,) = [widget for widget in page.content.select(pn.widgets.Select) if widget.name == "Run folder"]
    assert sorted(run_folder_select.options.values()) == [
        str(session_with_two_runs / "output_1"),
        str(session_with_two_runs / "output_2"),
    ]
