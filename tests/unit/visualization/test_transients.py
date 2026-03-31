import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import matplotlib.figure
import numpy as np
import pytest

from guppy.visualization.transients import visualize_peaks

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def timestamps():
    return np.array([0.0, 1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def z_score():
    return np.array([0.1, 0.5, 0.9, 0.4, 0.2])


@pytest.fixture
def peaks_index():
    return np.array([2])  # index of the maximum value (0.9)


@pytest.fixture
def peaks_result(timestamps, z_score, peaks_index):
    return visualize_peaks("my_title", "my_suptitle", z_score, timestamps, peaks_index)


# ---------------------------------------------------------------------------
# visualize_peaks
# ---------------------------------------------------------------------------


def test_visualize_peaks_returns_figure(peaks_result):
    fig, ax = peaks_result
    assert isinstance(fig, matplotlib.figure.Figure)


def test_visualize_peaks_returns_axes(peaks_result):
    fig, ax = peaks_result
    assert isinstance(ax, matplotlib.axes.Axes)


def test_visualize_peaks_ax_title(peaks_result):
    fig, ax = peaks_result
    assert ax.get_title() == "my_title"


def test_visualize_peaks_suptitle(peaks_result):
    fig, ax = peaks_result
    suptitle_texts = [text.get_text() for text in fig.texts]
    assert "my_suptitle" in suptitle_texts


def test_visualize_peaks_signal_line_has_correct_data(peaks_result, timestamps, z_score):
    fig, ax = peaks_result
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), timestamps)
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), z_score)


def test_visualize_peaks_markers_have_correct_data(peaks_result, timestamps, z_score, peaks_index):
    fig, ax = peaks_result
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), timestamps[peaks_index])
    np.testing.assert_array_equal(ax.lines[1].get_ydata(), z_score[peaks_index])


def test_visualize_peaks_empty_peaks_index(timestamps, z_score):
    empty_index = np.array([], dtype=int)
    fig, ax = visualize_peaks("title", "suptitle", z_score, timestamps, empty_index)
    assert len(ax.lines) == 2
    assert len(ax.lines[1].get_xdata()) == 0
