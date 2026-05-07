# Set CI env var before any guppy imports to prevent the module-level
# plt.switch_backend("TKAgg") call in artifact_removal.py from running.
import os

os.environ["CI"] = "1"

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from guppy.frontend.artifact_removal import ArtifactRemovalWidget


class FakeEvent:
    def __init__(self, key=" ", xdata=0.0, ydata=0.0):
        self.key = key
        self.xdata = xdata
        self.ydata = ydata


@pytest.fixture
def widget(tmp_path):
    x = np.linspace(0, 10, 100)
    y1 = np.zeros(100)
    y2 = np.sin(x)
    y3 = np.cos(x)
    return ArtifactRemovalWidget(
        filepath=str(tmp_path),
        x=x,
        y1=y1,
        y2=y2,
        y3=y3,
        plot_name=["signal_DMS", "control_DMS", "fitted_DMS"],
        removeArtifacts=False,
    )


class TestArtifactRemovalWidget:
    def test_constructs_without_raising(self, widget):
        assert widget.coords == []

    def test_space_keypress_appends_coord(self, widget):
        widget._on_key_press(FakeEvent(key=" ", xdata=1.5))
        assert widget.coords == [(1.5, 0.0)]

    def test_two_space_keypresses_appends_two_coords(self, widget):
        widget._on_key_press(FakeEvent(key=" ", xdata=1.5))
        widget._on_key_press(FakeEvent(key=" ", xdata=3.0))
        assert widget.coords == [(1.5, 0.0), (3.0, 0.0)]

    def test_d_keypress_removes_last_coord(self, widget):
        widget._on_key_press(FakeEvent(key=" ", xdata=1.5))
        widget._on_key_press(FakeEvent(key=" ", xdata=3.0))
        widget._on_key_press(FakeEvent(key="d", xdata=0.0))
        assert widget.coords == [(1.5, 0.0)]

    def test_d_keypress_on_empty_coords_does_not_raise(self, widget):
        widget._on_key_press(FakeEvent(key="d", xdata=0.0))
        assert widget.coords == []

    def test_on_close_saves_npy_file(self, widget, tmp_path):
        widget._on_key_press(FakeEvent(key=" ", xdata=1.5))
        widget._on_key_press(FakeEvent(key=" ", xdata=3.0))
        widget._on_close(FakeEvent())

        expected_file = tmp_path / "coordsForPreProcessing_DMS.npy"
        assert expected_file.exists()
        saved = np.load(str(expected_file))
        assert saved.shape == (2, 2)

    def test_on_close_with_empty_coords_writes_no_file(self, widget, tmp_path):
        widget._on_close(FakeEvent())
        assert not (tmp_path / "coordsForPreProcessing_DMS.npy").exists()
