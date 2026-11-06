import holoviews as hv
import pytest

from guppy.visualization.palette import (
    CONTROL,
    COVARIATE,
    EPOCH_COLORS,
    HIGHLIGHT,
    SIGNAL,
    apply_plot_defaults,
)


@pytest.fixture
def holoviews_bokeh():
    hv.extension("bokeh")


class TestRoles:
    def test_signal_is_green_control_blue_covariates_violet_highlights_red(self):
        assert (SIGNAL, CONTROL, COVARIATE, HIGHLIGHT) == ("#3A9E5C", "#1F6FD1", "#7B4FD0", "#D62728")

    def test_epochs_run_along_the_logo_gradient_from_violet_to_yellow_green(self):
        assert EPOCH_COLORS == ["#6E56CF", "#1F7FC4", "#14A39A", "#9BC53D"]


class TestGuppyCycle:
    def test_the_palette_is_offered_by_name_starting_with_the_signal_green(self):
        assert hv.Cycle("GuPPy").values == [
            "#3A9E5C",
            "#1F6FD1",
            "#7B4FD0",
            "#D08C15",
            "#1598A0",
            "#D62728",
            "#6B7785",
        ]


class TestApplyPlotDefaults:
    def test_a_single_curve_takes_the_signal_green(self, holoviews_bokeh):
        apply_plot_defaults()

        figure = hv.render(hv.Curve([1, 2, 3]))

        assert figure.renderers[0].glyph.line_color == "#3A9E5C"

    def test_an_overlay_steps_through_the_palette(self, holoviews_bokeh):
        apply_plot_defaults()

        figure = hv.render(hv.NdOverlay({index: hv.Curve([index, index + 1]) for index in range(3)}))

        assert [renderer.glyph.line_color for renderer in figure.renderers] == ["#3A9E5C", "#1F6FD1", "#7B4FD0"]

    def test_an_explicit_color_wins(self, holoviews_bokeh):
        apply_plot_defaults()

        figure = hv.render(hv.Curve([1, 2]).opts(color="#123456"))

        assert figure.renderers[0].glyph.line_color == "#123456"
