import panel as pn
import pytest

from guppy.frontend.visualization_dashboard import VisualizationDashboard

BASENAME = "test_session"


@pytest.fixture(scope="session")
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
        # Panel wraps each reactive method in a ParamMethod pane. Verify that the
        # three panes corresponding to contPlot, update_selector, and plot_specific_trials
        # are present by inspecting the .object attribute of each ParamMethod.
        objects = dashboard._psth_tab.objects
        wrapped_methods = [obj.object for obj in objects if hasattr(obj, "object") and callable(obj.object)]
        assert plotter.contPlot in wrapped_methods
        assert plotter.update_selector in wrapped_methods
        assert plotter.plot_specific_trials in wrapped_methods

    def test_heatmap_tab_is_panel_column(self, dashboard):
        assert isinstance(dashboard._heatmap_tab, pn.Column)

    def test_heatmap_tab_title_contains_basename(self, dashboard):
        first_object = dashboard._heatmap_tab.objects[0]
        content = first_object.object if hasattr(first_object, "object") else str(first_object)
        assert BASENAME in content

    def test_heatmap_tab_embeds_heatmap(self, dashboard, plotter):
        # Panel wraps the heatmap reactive method in a ParamMethod inside a pn.Row.
        heatmap_row = dashboard._heatmap_tab.objects[-1]
        assert isinstance(heatmap_row, pn.Row)
        wrapped_methods = [obj.object for obj in heatmap_row.objects if hasattr(obj, "object") and callable(obj.object)]
        assert plotter.heatmap in wrapped_methods
