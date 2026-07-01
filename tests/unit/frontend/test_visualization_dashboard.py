import panel as pn
import pytest

from guppy.frontend.visualization_dashboard import VisualizationDashboard

BASENAME = "test_session"


@pytest.fixture
def dashboard(plotter, panel_extension):
    return VisualizationDashboard(plotter=plotter, basename=BASENAME)


class TestVisualizationDashboard:
    def test_constructs(self, dashboard, plotter):
        assert dashboard.plotter is plotter
        assert dashboard.basename == BASENAME

    def test_psth_tab_is_panel_column(self, dashboard):
        assert isinstance(dashboard._psth_tab, pn.Column)

    def test_psth_tab_title_contains_basename(self, dashboard):
        first_object = dashboard._psth_tab.objects[0]
        content = first_object.object if hasattr(first_object, "object") else str(first_object)
        assert BASENAME in content

    def test_psth_tab_embeds_reactive_plots(self, dashboard, plotter):
        # Panel wraps each reactive method in a ParamMethod pane. Each plot now lives
        # inside its own Card, so search the tab recursively for the three panes.
        panes = dashboard._psth_tab.select(
            lambda obj: hasattr(obj, "object") and callable(getattr(obj, "object", None))
        )
        wrapped_methods = [pane.object for pane in panes]
        assert plotter.contPlot in wrapped_methods
        assert plotter.update_selector in wrapped_methods
        assert plotter.plot_specific_trials in wrapped_methods

    def test_heatmap_tab_is_panel_column(self, dashboard):
        assert isinstance(dashboard._heatmap_tab, pn.Column)

    def test_heatmap_tab_title_contains_basename(self, dashboard):
        first_object = dashboard._heatmap_tab.objects[0]
        content = first_object.object if hasattr(first_object, "object") else str(first_object)
        assert BASENAME in content

    def test_psth_tab_uses_number_inputs_for_axis_limits(self, dashboard):
        # 3 plots x 2 axes x (min, max) = 12 FloatInput boxes.
        number_inputs = list(dashboard._psth_tab.select(pn.widgets.FloatInput))
        assert len(number_inputs) == 12

    def test_psth_tab_has_trace_and_mean_color_pickers(self, dashboard):
        color_pickers = list(dashboard._psth_tab.select(pn.widgets.ColorPicker))
        assert len(color_pickers) == 2

    def test_psth_tab_trial_picker_is_cross_selector(self, dashboard):
        # Both the event picker and the trial picker are CrossSelectors.
        cross_selectors = list(dashboard._psth_tab.select(pn.widgets.CrossSelector))
        assert len(cross_selectors) == 2

    def test_save_controls_expose_format_and_button(self, dashboard):
        row = dashboard._save_controls(options_name="save_options_cont", action_name="save_cont")
        assert list(row.select(pn.widgets.Select))
        assert list(row.select(pn.widgets.Button))

    def test_range_number_inputs_param_updates_boxes(self, dashboard):
        row = dashboard._range_number_inputs(name="cont_X", label="X")
        minimum, maximum = row[0], row[1]

        dashboard.plotter.cont_X = (1.0, 2.0)

        assert minimum.value == 1.0
        assert maximum.value == 2.0

    def test_range_number_inputs_boxes_update_param(self, dashboard):
        row = dashboard._range_number_inputs(name="cont_Y", label="Y")
        minimum, maximum = row[0], row[1]

        minimum.value = -3.0
        maximum.value = 4.0

        assert dashboard.plotter.cont_Y == (-3.0, 4.0)

    def test_heatmap_tab_embeds_heatmap(self, dashboard, plotter):
        # Panel wraps the heatmap reactive method in a ParamMethod inside a pn.Row.
        heatmap_row = dashboard._heatmap_tab.objects[-1]
        assert isinstance(heatmap_row, pn.Row)
        wrapped_methods = [obj.object for obj in heatmap_row.objects if hasattr(obj, "object") and callable(obj.object)]
        assert plotter.heatmap in wrapped_methods
