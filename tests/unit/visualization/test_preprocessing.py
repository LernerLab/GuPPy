import os

os.environ["CI"] = "1"  # Must precede guppy.visualization.preprocessing import (prevents TKAgg switch)

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import matplotlib.figure
import numpy as np
import pytest

from guppy.visualization.preprocessing import (
    visualize_control_signal_fit,
    visualize_preprocessing,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def timestamps():
    return np.array([0.0, 1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def signal():
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5])


# Constant arrays make it trivial to verify which data lands on which axes.
@pytest.fixture
def x():
    return np.array([0.0, 1.0, 2.0])


@pytest.fixture
def y1():
    return np.ones(3) * 1.0


@pytest.fixture
def y2():
    return np.ones(3) * 2.0


@pytest.fixture
def y3():
    return np.ones(3) * 3.0


@pytest.fixture
def plot_name():
    return ["control", "signal", "fit"]


@pytest.fixture
def name():
    return "test_session"


@pytest.fixture
def preprocessing_result(timestamps, signal):
    return visualize_preprocessing(suptitle="my_suptitle", title="my_title", x=timestamps, y=signal)


@pytest.fixture
def control_signal_fit_result(x, y1, y2, y3, plot_name, name):
    return visualize_control_signal_fit(x, y1, y2, y3, plot_name, name, False)


@pytest.fixture
def control_signal_fit_result_artifacts_removed(x, y1, y2, y3, plot_name, name):
    return visualize_control_signal_fit(x, y1, y2, y3, plot_name, name, True)


# ---------------------------------------------------------------------------
# visualize_preprocessing
# ---------------------------------------------------------------------------


def test_visualize_preprocessing_returns_figure(preprocessing_result):
    fig, ax = preprocessing_result
    assert isinstance(fig, matplotlib.figure.Figure)


def test_visualize_preprocessing_returns_axes(preprocessing_result):
    fig, ax = preprocessing_result
    assert isinstance(ax, matplotlib.axes.Axes)


def test_visualize_preprocessing_ax_title(preprocessing_result):
    fig, ax = preprocessing_result
    assert ax.get_title() == "my_title"


def test_visualize_preprocessing_suptitle(preprocessing_result):
    fig, ax = preprocessing_result
    suptitle_texts = [text.get_text() for text in fig.texts]
    assert "my_suptitle" in suptitle_texts


def test_visualize_preprocessing_plots_correct_x_data(preprocessing_result, timestamps):
    fig, ax = preprocessing_result
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), timestamps)


def test_visualize_preprocessing_plots_correct_y_data(preprocessing_result, signal):
    fig, ax = preprocessing_result
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), signal)


# ---------------------------------------------------------------------------
# visualize_control_signal_fit
# ---------------------------------------------------------------------------


def test_visualize_control_signal_fit_returns_figure(control_signal_fit_result):
    fig, ax1, ax2, ax3 = control_signal_fit_result
    assert isinstance(fig, matplotlib.figure.Figure)


def test_visualize_control_signal_fit_returns_three_axes(control_signal_fit_result):
    fig, ax1, ax2, ax3 = control_signal_fit_result
    assert isinstance(ax1, matplotlib.axes.Axes)
    assert isinstance(ax2, matplotlib.axes.Axes)
    assert isinstance(ax3, matplotlib.axes.Axes)


def test_visualize_control_signal_fit_ax1_title(control_signal_fit_result, plot_name):
    fig, ax1, ax2, ax3 = control_signal_fit_result
    assert ax1.get_title() == plot_name[0]


def test_visualize_control_signal_fit_ax2_title(control_signal_fit_result, plot_name):
    fig, ax1, ax2, ax3 = control_signal_fit_result
    assert ax2.get_title() == plot_name[1]


def test_visualize_control_signal_fit_ax3_title(control_signal_fit_result, plot_name):
    fig, ax1, ax2, ax3 = control_signal_fit_result
    assert ax3.get_title() == plot_name[2]


def test_visualize_control_signal_fit_suptitle(control_signal_fit_result, name):
    fig, ax1, ax2, ax3 = control_signal_fit_result
    suptitle_texts = [text.get_text() for text in fig.texts]
    assert name in suptitle_texts


def test_visualize_control_signal_fit_ax1_plots_y1(control_signal_fit_result, y1):
    fig, ax1, ax2, ax3 = control_signal_fit_result
    np.testing.assert_array_equal(ax1.lines[0].get_ydata(), y1)


def test_visualize_control_signal_fit_ax2_plots_y2(control_signal_fit_result, y2):
    fig, ax1, ax2, ax3 = control_signal_fit_result
    np.testing.assert_array_equal(ax2.lines[0].get_ydata(), y2)


def test_visualize_control_signal_fit_ax3_has_two_lines(control_signal_fit_result):
    fig, ax1, ax2, ax3 = control_signal_fit_result
    assert len(ax3.lines) == 2


def test_visualize_control_signal_fit_ax3_plots_y2_then_y3(control_signal_fit_result, y2, y3):
    fig, ax1, ax2, ax3 = control_signal_fit_result
    np.testing.assert_array_equal(ax3.lines[0].get_ydata(), y2)
    np.testing.assert_array_equal(ax3.lines[1].get_ydata(), y3)


def test_visualize_control_signal_fit_xlabel_without_artifacts(control_signal_fit_result):
    fig, ax1, ax2, ax3 = control_signal_fit_result
    assert ax3.get_xlabel() == "Time(s)"


def test_visualize_control_signal_fit_xlabel_with_artifacts_removed(control_signal_fit_result_artifacts_removed):
    fig, ax1, ax2, ax3 = control_signal_fit_result_artifacts_removed
    assert "Note" in ax3.get_xlabel()
