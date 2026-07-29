import holoviews as hv
import numpy as np
import pytest

from guppy.visualization.preprocessing import (
    build_control_signal_fit,
    build_preprocessing_curve,
)


@pytest.fixture
def timestamps():
    return np.array([0.0, 1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def signal():
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5])


class TestBuildPreprocessingCurve:
    def test_returns_curve(self, panel_extension, timestamps, signal):
        curve = build_preprocessing_curve(suptitle="s", title="z_score_DMS", x=timestamps, y=signal)
        assert isinstance(curve, hv.Curve)

    def test_plots_correct_data(self, panel_extension, timestamps, signal):
        curve = build_preprocessing_curve(suptitle="s", title="z_score_DMS", x=timestamps, y=signal)
        np.testing.assert_array_equal(curve.dimension_values(0), timestamps)
        np.testing.assert_array_equal(curve.dimension_values(1), signal)


class TestBuildControlSignalFit:
    @pytest.fixture
    def control(self):
        return np.ones(5) * 1.0

    @pytest.fixture
    def signal_trace(self):
        return np.ones(5) * 2.0

    @pytest.fixture
    def fit(self):
        return np.ones(5) * 3.0

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
        # Top two are bare curves; the bottom overlays the signal and the fit.
        control_curve, signal_curve, fit_overlay = layout.values()
        np.testing.assert_array_equal(control_curve.dimension_values(1), control)
        np.testing.assert_array_equal(signal_curve.dimension_values(1), signal_trace)
        assert isinstance(fit_overlay, hv.Overlay)
        fit_curves = fit_overlay.values()
        np.testing.assert_array_equal(fit_curves[0].dimension_values(1), signal_trace)
        np.testing.assert_array_equal(fit_curves[1].dimension_values(1), fit)

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
        for element in layout.values():
            assert isinstance(element, hv.Overlay)
            # Two windows -> two VSpans shaded on this axis.
            vspans = [item for item in element.values() if isinstance(item, hv.VSpan)]
            assert len(vspans) == 2
