import holoviews as hv
import numpy as np
import panel as pn
import pytest

from guppy.frontend.raw_data_view import (
    RawDataView,
    build_raw_data_view_page,
    load_raw_data,
)
from guppy.utils._hdf5_io import write_hdf5
from guppy.utils.stores_list import write_stores_list
from guppy_test_data import resolve_plot


@pytest.fixture
def block_run_folder(tmp_path):
    """A run folder laid out as Step 2 writes block-structured (TDT) stores.

    Each channel holds two blocks of two samples at 4 Hz starting at 10.0 s and 10.5 s,
    so its sample times are 10.0, 10.25, 10.5, 10.75.
    """
    folder = tmp_path / "session_a" / "output_1"
    folder.mkdir(parents=True)
    write_stores_list(
        run_folder=folder,
        store_array=np.array(
            [["Dv1A", "Dv2A", "PrtN", "Spd1"], ["control_DMS", "signal_DMS", "port_entries", "covariate_speed"]]
        ),
    )
    for store_id, data in (("Dv1A", [1.0, 2.0, 3.0, 4.0]), ("Dv2A", [5.0, 6.0, 7.0])):
        write_hdf5(np.array([10.0, 10.5]), store_id, str(folder), "timestamps")
        write_hdf5(np.array(data), store_id, str(folder), "data")
        write_hdf5(2, store_id, str(folder), "npoints")
        write_hdf5(4.0, store_id, str(folder), "sampling_rate")
    write_hdf5(np.array([10.25, 10.5]), "PrtN", str(folder), "timestamps")
    write_hdf5(np.array([10.0, 11.0]), "Spd1", str(folder), "timestamps")
    write_hdf5(np.array([0.0, 1.0]), "Spd1", str(folder), "data")
    return folder


@pytest.fixture
def sample_run_folder(tmp_path):
    """A run folder laid out as Step 2 writes per-sample stores (CSV and the rest).

    Recording site B has a signal and no control, as when the isosbestic control is off.
    """
    folder = tmp_path / "session_b" / "output_1"
    folder.mkdir(parents=True)
    write_stores_list(
        run_folder=folder,
        store_array=np.array(
            [["sig_b", "sig_a", "ctrl_a", "ttl", "lick"], ["signal_B", "signal_A", "control_A", "reward", "lick"]]
        ),
    )
    write_hdf5(np.array([2.0, 3.0, 4.0]), "ctrl_a", str(folder), "timestamps")
    write_hdf5(np.array([0.1, 0.2, 0.3]), "ctrl_a", str(folder), "data")
    write_hdf5(np.array([2.0, 3.0, 4.0]), "sig_a", str(folder), "timestamps")
    write_hdf5(np.array([1.1, 1.2, 1.3]), "sig_a", str(folder), "data")
    write_hdf5(np.array([1.0, 2.0]), "sig_b", str(folder), "timestamps")
    write_hdf5(np.array([9.0, 8.0]), "sig_b", str(folder), "data")
    write_hdf5(np.array([2.5]), "ttl", str(folder), "timestamps")
    write_hdf5(np.array([1.5, 3.5]), "lick", str(folder), "timestamps")
    return folder


class TestLoadRawData:
    def test_block_timestamps_expand_to_one_per_sample_from_the_first_sample(self, block_run_folder):
        channels, _ = load_raw_data(run_folder=str(block_run_folder))

        assert list(channels) == ["DMS"]
        assert list(channels["DMS"]) == ["control_DMS", "signal_DMS"]
        control_times, control_values = channels["DMS"]["control_DMS"]
        np.testing.assert_allclose(control_times, [0.0, 0.25, 0.5, 0.75])
        np.testing.assert_array_equal(control_values, [1.0, 2.0, 3.0, 4.0])

    def test_a_channel_shorter_than_its_timestamps_is_cut_to_its_samples(self, block_run_folder):
        channels, _ = load_raw_data(run_folder=str(block_run_folder))

        signal_times, signal_values = channels["DMS"]["signal_DMS"]
        np.testing.assert_allclose(signal_times, [0.0, 0.25, 0.5])
        np.testing.assert_array_equal(signal_values, [5.0, 6.0, 7.0])

    def test_events_shift_with_the_channels_and_covariates_are_left_out(self, block_run_folder):
        _, event_times = load_raw_data(run_folder=str(block_run_folder))

        assert list(event_times) == ["port_entries"]
        np.testing.assert_allclose(event_times["port_entries"], [0.25, 0.5])

    def test_per_sample_stores_group_by_site_from_the_earliest_channel_sample(self, sample_run_folder):
        channels, event_times = load_raw_data(run_folder=str(sample_run_folder))

        assert {site: list(site_channels) for site, site_channels in channels.items()} == {
            "A": ["control_A", "signal_A"],
            "B": ["signal_B"],
        }
        np.testing.assert_allclose(channels["A"]["signal_A"][0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(channels["A"]["signal_A"][1], [1.1, 1.2, 1.3])
        np.testing.assert_allclose(channels["B"]["signal_B"][0], [0.0, 1.0])
        assert list(event_times) == ["lick", "reward"]
        np.testing.assert_allclose(event_times["lick"], [0.5, 2.5])
        np.testing.assert_allclose(event_times["reward"], [1.5])


def _panel_titles(view):
    return [resolve_plot(axis).opts.get("plot").kwargs["title"] for axis in view.plot_pane.object.values()]


def _event_lines(view, index):
    """Event label → x positions of the vertical lines drawn on one panel."""
    plot = resolve_plot(view.plot_pane.object.values()[index])
    elements = list(plot.values()) if isinstance(plot, hv.Overlay) else [plot]
    return {item.label: list(item.data["time (s)"]) for item in elements if isinstance(item, hv.VLines)}


@pytest.fixture
def raw_data_view(panel_extension, sample_run_folder):
    channels, event_times = load_raw_data(run_folder=str(sample_run_folder))
    return RawDataView(channels=channels, event_times=event_times, label="output_1")


class TestRawDataView:
    def test_stacks_the_first_sites_channels(self, raw_data_view):
        assert raw_data_view.site_select.options == ["A", "B"]
        assert _panel_titles(raw_data_view) == ["output_1 — control_A", "signal_A"]

    def test_site_selector_switches_the_channels(self, raw_data_view):
        raw_data_view.site_select.value = "B"
        assert _panel_titles(raw_data_view) == ["output_1 — signal_B"]

    def test_marks_every_event_on_every_panel(self, raw_data_view):
        assert raw_data_view.event_select.value == ["lick", "reward"]
        for index in range(2):
            assert _event_lines(raw_data_view, index) == {"lick": [0.5, 2.5], "reward": [1.5]}

    def test_unchecking_an_event_removes_its_lines(self, raw_data_view):
        raw_data_view.event_select.value = ["reward"]
        assert _event_lines(raw_data_view, 0) == {"reward": [1.5]}

    def test_no_event_stores_shows_no_event_picker(self, panel_extension):
        view = RawDataView(
            channels={"A": {"signal_A": (np.array([0.0, 1.0]), np.array([1.0, 2.0]))}},
            event_times={},
            label="output_1",
        )
        assert "**Events**" not in [item.object for item in view.widget if isinstance(item, pn.pane.Markdown)]
        assert _event_lines(view, 0) == {}


class TestBuildRawDataViewPage:
    def test_single_folder_has_no_folder_selector(self, panel_extension, sample_run_folder):
        page = build_raw_data_view_page(run_folders=[str(sample_run_folder)])
        assert [widget for widget in page.select(pn.widgets.Select) if widget.name == "Run folder"] == []

    def test_multiple_folders_get_a_folder_selector(self, panel_extension, block_run_folder, sample_run_folder):
        page = build_raw_data_view_page(run_folders=[str(block_run_folder), str(sample_run_folder)])
        (run_folder_select,) = [widget for widget in page.select(pn.widgets.Select) if widget.name == "Run folder"]
        assert set(run_folder_select.options.values()) == {str(block_run_folder), str(sample_run_folder)}
