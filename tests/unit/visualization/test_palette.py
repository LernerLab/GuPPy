import holoviews as hv
import pytest

from guppy.visualization.palette import (
    CONTROL,
    COVARIATE,
    EPOCH_COLORS,
    HIGHLIGHT,
    PLOT_THEME,
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


class TestPlotText:
    def test_axis_text_is_helvetica_with_16px_labels_and_14px_ticks(self, holoviews_bokeh):
        apply_plot_defaults()

        axis = hv.render(hv.Curve([1, 2, 3])).xaxis[0]

        assert (axis.axis_label_text_font, axis.axis_label_text_font_size, axis.major_label_text_font_size) == (
            "Helvetica, Arial, sans-serif",
            "16px",
            "14px",
        )

    def test_axis_labels_and_titles_are_upright(self):
        attributes = PLOT_THEME._json["attrs"]

        assert (attributes["Axis"]["axis_label_text_font_style"], attributes["Title"]["text_font_style"]) == (
            "normal",
            "normal",
        )

    def test_legends_use_the_tick_size(self):
        assert PLOT_THEME._json["attrs"]["Legend"]["label_text_font_size"] == "14px"
