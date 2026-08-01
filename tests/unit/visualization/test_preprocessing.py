import holoviews as hv
import numpy as np
import pytest

from guppy.visualization.preprocessing import (
    build_control_signal_fit,
    build_preprocessing_curve,
)
from guppy_test_data import resolve_plot


def image_extent(image):
    """Return an image's ``(left, bottom, right, top)`` bounds as plain floats.

    A shaded trace is an ``hv.RGB``, so its data is pixels rather than samples; the bounds
    are what carry the extent of the trace that was drawn into it.
    """
    return tuple(float(value) for value in image.bounds.lbrt())


@pytest.fixture
def timestamps():
    return np.array([0.0, 1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def signal():
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5])


class TestBuildPreprocessingCurve:
    def test_returns_shaded_image(self, panel_extension, timestamps, signal):
        curve = build_preprocessing_curve(suptitle="s", title="z_score_DMS", x=timestamps, y=signal)
        assert isinstance(resolve_plot(curve), hv.RGB)

    def test_plots_correct_data(self, panel_extension, timestamps, signal):
        curve = resolve_plot(build_preprocessing_curve(suptitle="s", title="z_score_DMS", x=timestamps, y=signal))
        assert image_extent(curve) == (0.0, 0.1, 4.0, 0.5)


class TestBuildControlSignalFit:
    # Each trace spans a distinct range so the bounds of a shaded panel identify which
    # trace was drawn into it.
    @pytest.fixture
    def control(self):
        return np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    @pytest.fixture
    def signal_trace(self):
        return np.array([10.0, 11.0, 12.0, 13.0, 14.0])

    @pytest.fixture
    def fit(self):
        return np.array([20.0, 21.0, 22.0, 23.0, 24.0])

    def test_returns_three_stacked_elements(self, panel_extension, timestamps, control, signal_trace, fit):
        layout = build_control_signal_fit(
            x=timestamps,
            control=control,
            signal=signal_trace,
            fit=fit,
            titles=["control", "signal", "fit"],
            suptitle="session",
            artifacts_have_been_removed=False,
        )
        assert isinstance(layout, hv.Layout)
        elements = layout.values()
        assert len(elements) == 3

    def test_bottom_axis_overlays_signal_and_fit(self, panel_extension, timestamps, control, signal_trace, fit):
        layout = build_control_signal_fit(
            x=timestamps,
            control=control,
            signal=signal_trace,
            fit=fit,
            titles=["control", "signal", "fit"],
            suptitle="session",
            artifacts_have_been_removed=False,
        )
        # Top two are single shaded traces; the bottom overlays the signal and the fit.
        control_image, signal_image, fit_overlay = (resolve_plot(plot) for plot in layout.values())
        assert image_extent(control_image) == (0.0, 1.0, 4.0, 5.0)
        assert image_extent(signal_image) == (0.0, 10.0, 4.0, 14.0)

        assert isinstance(fit_overlay, hv.Overlay)
        shaded_signal, shaded_fit = fit_overlay.values()
        assert image_extent(shaded_signal) == (0.0, 10.0, 4.0, 14.0)
        assert image_extent(shaded_fit) == (0.0, 20.0, 4.0, 24.0)

    def test_windows_shade_all_three_axes(self, panel_extension, timestamps, control, signal_trace, fit):
        layout = build_control_signal_fit(
            x=timestamps,
            control=control,
            signal=signal_trace,
            fit=fit,
            titles=["control", "signal", "fit"],
            suptitle="session",
            artifacts_have_been_removed=False,
            windows=[(1.0, 2.0), (3.0, 4.0)],
        )
        for plot in layout.values():
            element = resolve_plot(plot)
            assert isinstance(element, hv.Overlay)
            # Two windows -> two VSpans shaded on this axis.
            vspans = [item for item in element.values() if isinstance(item, hv.VSpan)]
            assert len(vspans) == 2
