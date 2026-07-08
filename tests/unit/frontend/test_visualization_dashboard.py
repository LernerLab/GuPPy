import panel as pn
import pytest

from guppy.frontend.visualization_dashboard import VisualizationDashboard

BASENAME = "test_session"


class _FakeRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _FakeFigure:
    def __init__(self):
        self.x_range = _FakeRange(0.0, 1.0)
        self.y_range = _FakeRange(-1.0, 1.0)


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
        # trace_color and mean_color are exposed by name; per-event pickers add a
        # dynamic number on top, so assert these two static pickers are present
        # rather than a fixed total count.
        picker_names = {picker.name for picker in dashboard._psth_tab.select(pn.widgets.ColorPicker)}
        assert {"Trace color", "Mean color"} <= picker_names

    def test_per_event_color_pickers_card_is_collapsed(self, dashboard):
        card = dashboard._per_event_color_pickers()
        assert isinstance(card, pn.Card)
        assert card.collapsed is True

    def test_per_event_color_pickers_one_per_selected_event(self, dashboard):
        dashboard.plotter.selector_for_multipe_events_plot = ["event1", "event2"]
        card = dashboard._per_event_color_pickers()
        # Each event is a Row of [ColorPicker, Markdown label]; the label carries the name.
        labels = [pane.object for pane in card.select(pn.pane.Markdown)]
        assert labels == ["event1", "event2"]
        assert len(list(card.select(pn.widgets.ColorPicker))) == 2

    def test_per_event_color_picker_value_writes_override(self, dashboard):
        dashboard.plotter.selector_for_multipe_events_plot = ["event1", "event2"]
        card = dashboard._per_event_color_pickers()
        pickers = list(card.select(pn.widgets.ColorPicker))
        labels = [pane.object for pane in card.select(pn.pane.Markdown)]
        pickers_by_event = dict(zip(labels, pickers))

        pickers_by_event["event2"].value = "#123456"

        assert dashboard.plotter.overlay_color_overrides == {"event2": "#123456"}

    def test_per_event_color_pickers_reset_button_clears_overrides(self, dashboard):
        dashboard.plotter.selector_for_multipe_events_plot = ["event1", "event2"]
        dashboard.plotter.overlay_color_overrides = {"event1": "#111111"}
        card = dashboard._per_event_color_pickers()
        reset_button = next(iter(card.select(pn.widgets.Button)))

        reset_button.clicks += 1

        assert dashboard.plotter.overlay_color_overrides == {}

    def test_per_event_color_pickers_rebuild_on_selection_change(self, dashboard):
        card = dashboard._per_event_color_pickers()
        dashboard.plotter.selector_for_multipe_events_plot = ["event1", "event2"]
        labels = [pane.object for pane in card.select(pn.pane.Markdown)]
        assert labels == ["event1", "event2"]

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

    def test_range_number_inputs_box_edit_moves_live_figure(self, dashboard):
        # End-to-end: typing a box value moves the already-rendered figure in place.
        figure = _FakeFigure()
        dashboard.plotter._figures["cont"] = figure
        row = dashboard._range_number_inputs(name="cont_X", label="X")
        minimum, maximum = row[0], row[1]

        minimum.value = 0.0
        maximum.value = 3.0

        assert (figure.x_range.start, figure.x_range.end) == (0.0, 3.0)

    def test_psth_tab_has_hide_minor_ticks_checkbox(self, dashboard, plotter):
        checkboxes = {box.name: box for box in dashboard._psth_tab.select(pn.widgets.Checkbox)}
        assert "Hide minor tick marks" in checkboxes

        checkbox = checkboxes["Hide minor tick marks"]
        assert checkbox.value is False  # ticks shown by default

        checkbox.value = True
        assert plotter.hide_minor_ticks is True

    def test_heatmap_tab_embeds_heatmap(self, dashboard, plotter):
        # The heatmap reactive method is wrapped in a ParamMethod pane nested inside the
        # heatmap Card, so search the tab recursively (as the PSTH tab test does).
        panes = dashboard._heatmap_tab.select(
            lambda obj: hasattr(obj, "object") and callable(getattr(obj, "object", None))
        )
        assert plotter.heatmap in [pane.object for pane in panes]

    def test_heatmap_tab_uses_number_inputs_for_axis_and_color_limits(self, dashboard):
        # X + colour-scale = 2 ranges x (min, max) = 4 FloatInput boxes. (The Trials
        # (Y) axis is fixed and has no controls.)
        number_inputs = list(dashboard._heatmap_tab.select(pn.widgets.FloatInput))
        assert len(number_inputs) == 4

    def test_heatmap_tab_has_hide_minor_ticks_checkbox(self, dashboard, plotter):
        checkboxes = {box.name: box for box in dashboard._heatmap_tab.select(pn.widgets.Checkbox)}
        assert "Hide minor tick marks" in checkboxes

        checkbox = checkboxes["Hide minor tick marks"]
        assert checkbox.value is False  # ticks shown by default

        checkbox.value = True
        assert plotter.hide_minor_ticks_heatmap is True

    def test_heatmap_tab_has_ticks_inside_checkbox(self, dashboard, plotter):
        checkboxes = {box.name: box for box in dashboard._heatmap_tab.select(pn.widgets.Checkbox)}
        assert "Ticks inside" in checkboxes

        checkbox = checkboxes["Ticks inside"]
        assert checkbox.value is False  # ticks point outward by default

        checkbox.value = True
        assert plotter.ticks_inside_heatmap is True

    def test_heatmap_tab_has_hide_outer_border_checkbox(self, dashboard, plotter):
        checkboxes = {box.name: box for box in dashboard._heatmap_tab.select(pn.widgets.Checkbox)}
        assert "Hide top/right border" in checkboxes

        checkbox = checkboxes["Hide top/right border"]
        assert checkbox.value is False  # full border shown by default

        checkbox.value = True
        assert plotter.hide_outer_border_heatmap is True

    def test_heatmap_tab_has_save_controls(self, dashboard):
        assert list(dashboard._heatmap_tab.select(pn.widgets.Button))
        selects = {select.name for select in dashboard._heatmap_tab.select(pn.widgets.Select)}
        assert "Save format" in selects
