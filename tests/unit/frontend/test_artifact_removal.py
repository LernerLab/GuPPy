# Set CI env var before any guppy imports to prevent the module-level
# plt.switch_backend("TKAgg") call in artifact_removal.py from running.
import os

os.environ["CI"] = "1"

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def _make_stub_figure():
    """Return (fig, ax1, ax2, ax3) on the Agg backend."""
    fig, axes = plt.subplots(3, 1)
    return fig, axes[0], axes[1], axes[2]


def _make_widget(tmp_path, monkeypatch, remove_artifacts=False):
    """Construct an ArtifactRemovalWidget with synthetic data, patching visualize_control_signal_fit."""
    stub = _make_stub_figure()
    monkeypatch.setattr(
        "guppy.frontend.artifact_removal.visualize_control_signal_fit",
        lambda *args, **kwargs: stub,
    )
    from guppy.frontend.artifact_removal import ArtifactRemovalWidget

    x = np.linspace(0, 10, 100)
    y1 = np.zeros(100)
    y2 = np.sin(x)
    y3 = np.cos(x)
    plot_name = ["signal_DMS"]
    return ArtifactRemovalWidget(
        filepath=str(tmp_path),
        x=x,
        y1=y1,
        y2=y2,
        y3=y3,
        plot_name=plot_name,
        removeArtifacts=remove_artifacts,
    )


def _make_key_event(key, xdata, ydata=0.0):
    """Create a minimal mock key-press event object."""

    class FakeEvent:
        pass

    event = FakeEvent()
    event.key = key
    event.xdata = xdata
    event.ydata = ydata
    return event


def _make_close_event():
    class FakeEvent:
        pass

    return FakeEvent()


def test_artifact_removal_widget_constructs_without_raising(tmp_path, monkeypatch):
    widget = _make_widget(tmp_path, monkeypatch)
    assert widget.coords == []


def test_artifact_removal_widget_space_keypress_appends_coord(tmp_path, monkeypatch):
    widget = _make_widget(tmp_path, monkeypatch)
    widget._on_key_press(_make_key_event(" ", xdata=1.5))
    assert widget.coords == [(1.5, 0.0)]


def test_artifact_removal_widget_two_space_keypresses_appends_two_coords(tmp_path, monkeypatch):
    widget = _make_widget(tmp_path, monkeypatch)
    widget._on_key_press(_make_key_event(" ", xdata=1.5))
    widget._on_key_press(_make_key_event(" ", xdata=3.0))
    assert widget.coords == [(1.5, 0.0), (3.0, 0.0)]


def test_artifact_removal_widget_d_keypress_removes_last_coord(tmp_path, monkeypatch):
    widget = _make_widget(tmp_path, monkeypatch)
    widget._on_key_press(_make_key_event(" ", xdata=1.5))
    widget._on_key_press(_make_key_event(" ", xdata=3.0))
    widget._on_key_press(_make_key_event("d", xdata=0.0))
    assert widget.coords == [(1.5, 0.0)]


def test_artifact_removal_widget_d_keypress_on_empty_coords_does_not_raise(tmp_path, monkeypatch):
    widget = _make_widget(tmp_path, monkeypatch)
    widget._on_key_press(_make_key_event("d", xdata=0.0))
    assert widget.coords == []


def test_artifact_removal_widget_on_close_saves_npy_file(tmp_path, monkeypatch):
    widget = _make_widget(tmp_path, monkeypatch)
    widget._on_key_press(_make_key_event(" ", xdata=1.5))
    widget._on_key_press(_make_key_event(" ", xdata=3.0))

    # Manually trigger the close handler (simulates closing the figure)
    widget._on_close(_make_close_event())

    expected_file = tmp_path / "coordsForPreProcessing_DMS.npy"
    assert expected_file.exists()
    saved = np.load(str(expected_file))
    assert saved.shape == (2, 2)


def test_artifact_removal_widget_on_close_with_empty_coords_writes_no_file(tmp_path, monkeypatch):
    widget = _make_widget(tmp_path, monkeypatch)
    widget._on_close(_make_close_event())

    expected_file = tmp_path / "coordsForPreProcessing_DMS.npy"
    assert not expected_file.exists()
