import panel as pn
import pytest

from guppy.frontend.visualization_dashboard import VisualizationDashboard

BASENAME = "test_session"
EVENTS = ["event1", "event2"]


class _FakeRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _FakeFigure:
    def __init__(self):
        self.x_range = _FakeRange(0.0, 1.0)
        self.y_range = _FakeRange(-1.0, 1.0)


def metric_selectors(tab):
    """Return the metric selectors rendered inside one dashboard tab."""
    return [widget for widget in tab.select(pn.widgets.Select) if "z_score" in widget.values]


def field_labels(layout):
    """Return the text of the toolbar field labels shown inside a layout, in order."""
    labels = [pane.object for pane in layout.select(pn.pane.HTML) if "white-space: nowrap" in pane.object]
    return [label.split(">")[1].split("<")[0] for label in labels]


def header_text(card):
    """Return the HTML of a card's header, which carries its title and note."""
    return card.header.object


@pytest.fixture
def dashboard(plotter, panel_extension):
    return VisualizationDashboard(
        plotter=plotter,
        basename=BASENAME,
        events=EVENTS,
        metric="z_score",
        available_metrics=["z_score", "dff"],
    )


@pytest.fixture
def single_metric_dashboard(plotter, panel_extension):
    """A dashboard for an output directory where step 4 only computed the z-score."""
    return VisualizationDashboard(
        plotter=plotter,
        basename=BASENAME,
        events=EVENTS,
        metric="z_score",
        available_metrics=["z_score"],
    )


class TestVisualizationDashboard:
    def test_constructs(self, dashboard, plotter):
        assert dashboard.plotter is plotter
        assert dashboard.basename == BASENAME

    def test_metric_selector_offers_every_available_metric(self, dashboard):
        for tab in (dashboard._psth_tab, dashboard._heatmap_tab):
            (selector,) = metric_selectors(tab)
            assert selector.options == {"z-score": "z_score", "\u0394F/F": "dff"}
            assert selector.value == "z_score"

    def test_metric_selector_offers_the_one_metric_that_was_computed(self, single_metric_dashboard):
        for tab in (single_metric_dashboard._psth_tab, single_metric_dashboard._heatmap_tab):
            (selector,) = metric_selectors(tab)
            assert selector.options == {"z-score": "z_score"}
            assert selector.value == "z_score"
            assert selector.disabled is False

    def test_metric_selector_leads_each_tabs_display_settings(self, dashboard):
        for tab in (dashboard._psth_tab, dashboard._heatmap_tab):
            display_settings = tab[0]
            first_field = display_settings[0][0]

            assert "Display Settings" in header_text(display_settings)
            assert field_labels(first_field)[0] == "Metric"
            assert first_field[1][0] is metric_selectors(tab)[0]

    def test_cards_are_titled_in_title_case_without_numbers(self, dashboard):
        titles = [header_text(card) for card in [*dashboard._psth_tab, *dashboard._heatmap_tab]]

        for title, expected in zip(
            titles,
            [
                "Display Settings",
                "Single-Event PSTH",
                "Multi-Event Comparison",
                "Selected Trials",
                "Display Settings",
                "Trial Heat Map",
            ],
            strict=True,
        ):
            assert f">{expected}</span>" in title

    def test_the_y_axis_label_is_not_a_display_setting(self, dashboard):
        """The label follows the metric, so no widget offers to change it."""
        widget_names = [widget.name for widget in dashboard._psth_tab.select(pn.widgets.Select)]

        assert "Y Label" not in widget_names

    def test_the_tabs_are_the_whole_template(self, dashboard):
        template = dashboard.build_template()

        assert list(template.main.objects) == [dashboard._tabs]

    def test_changing_the_metric_rebuilds_only_the_psth_and_heatmap_tabs(self, dashboard, plotter, monkeypatch):
        requested = {}
        monkeypatch.setattr(
            "guppy.frontend.visualization_dashboard.build_plotter",
            lambda **kwargs: requested.update(kwargs) or plotter,
        )
        unchanged_tabs = list(dashboard._tabs.objects[2:])

        metric_selectors(dashboard._psth_tab)[0].value = "dff"

        assert requested["metric"] == "dff"
        assert requested["events"] == EVENTS
        assert requested["filepath"] == plotter.filepath
        assert dashboard.metric == "dff"
        assert dashboard._tabs._names == ["PSTH", "Heat Map", "Tonic", "Binned", "Covariates", "Significance"]
        assert list(dashboard._tabs.objects[2:]) == unchanged_tabs

    def test_changing_the_metric_in_one_tab_moves_the_other_tab_selector(self, dashboard, plotter, monkeypatch):
        monkeypatch.setattr("guppy.frontend.visualization_dashboard.build_plotter", lambda **kwargs: plotter)

        metric_selectors(dashboard._heatmap_tab)[0].value = "dff"

        assert metric_selectors(dashboard._psth_tab)[0].value == "dff"
        assert metric_selectors(dashboard._heatmap_tab)[0].value == "dff"

    def test_changing_the_metric_keeps_the_axis_bounds(self, dashboard, plotter, monkeypatch):
        requested = {}
        monkeypatch.setattr(
            "guppy.frontend.visualization_dashboard.build_plotter",
            lambda **kwargs: requested.update(kwargs) or plotter,
        )

        metric_selectors(dashboard._psth_tab)[0].value = "dff"

        assert requested["x_min"] == -5.0
        assert requested["x_max"] == 10.0

    def test_build_template_exposes_every_tab_in_order(self, dashboard):
        dashboard.build_template()

        assert dashboard._tabs._names == [
            "PSTH",
            "Heat Map",
            "Tonic",
            "Binned",
            "Covariates",
            "Significance",
        ]

    def test_significance_tab_reports_an_empty_state_for_a_session_without_results(self, dashboard):
        # The plotter fixture's directory holds no significance output, so the tab renders
        # its note rather than failing -- which is what lets it be added unconditionally.
        assert "No PSTH significance results" in dashboard._significance_tab[0].object

    def test_the_tabs_stretch_to_the_page_width(self, dashboard):
        assert dashboard._tabs.sizing_mode == "stretch_width"
        assert dashboard._psth_tab.sizing_mode == "stretch_width"
        assert dashboard._heatmap_tab.sizing_mode == "stretch_width"

    def test_each_card_opens_on_one_toolbar_row(self, dashboard):
        cards = [*dashboard._psth_tab, *dashboard._heatmap_tab]

        assert [card.sizing_mode for card in cards] == ["stretch_width"] * 6
        assert [type(card[0]) for card in cards] == [pn.Row] * 6

    def test_pickers_sit_beside_their_plots_at_the_plot_height(self, dashboard, plotter):
        # Height_Plot defaults to 300 and height_heatmap to 600; each picker leaves 30px for its label.
        event_picker, trial_picker = dashboard._psth_tab.select(pn.widgets.CrossSelector)
        (heatmap_trial_picker,) = dashboard._heatmap_tab.select(pn.widgets.MultiSelect)

        assert (event_picker.height, trial_picker.height, heatmap_trial_picker.height) == (270, 270, 570)

        plotter.Height_Plot = 400

        assert (event_picker.height, trial_picker.height) == (370, 370)

    def test_psth_tab_is_panel_column(self, dashboard):
        assert isinstance(dashboard._psth_tab, pn.Column)

    def test_psth_tab_opens_on_its_settings_without_a_title(self, dashboard):
        assert isinstance(dashboard._psth_tab.objects[0], pn.Card)

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

    def test_heatmap_tab_opens_on_its_card_without_a_title(self, dashboard):
        assert isinstance(dashboard._heatmap_tab.objects[0], pn.Card)

    def test_psth_tab_uses_number_inputs_for_axis_limits(self, dashboard):
        # 3 plots x 2 axes x (min, max) = 12 FloatInput boxes.
        number_inputs = list(dashboard._psth_tab.select(pn.widgets.FloatInput))
        assert len(number_inputs) == 12

    def test_psth_tab_has_trace_and_mean_color_pickers(self, dashboard):
        assert {"Trace Color", "Mean Color"} <= set(field_labels(dashboard._psth_tab))

    def test_per_event_color_pickers_start_hidden(self, dashboard):
        strip = dashboard._per_event_color_pickers()
        assert strip.visible is False

    def test_colors_toggle_opens_the_per_event_color_pickers(self, dashboard):
        (toggle,) = dashboard._psth_tab.select(pn.widgets.Toggle)
        (strip,) = [box for box in dashboard._psth_tab.select(pn.FlexBox) if box.select(pn.widgets.Button)]

        toggle.value = True

        assert toggle.name == "Colors"
        assert strip.visible is True

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
        pickers_by_event = dict(zip(labels, pickers, strict=True))

        pickers_by_event["event2"].value = "#123456"

        assert dashboard.plotter.overlay_color_overrides == {"event2": "#123456"}

    def test_per_event_color_pickers_reset_button_clears_overrides(self, dashboard):
        dashboard.plotter.selector_for_multipe_events_plot = ["event1", "event2"]
        dashboard.plotter.overlay_color_overrides = {"event1": "#111111"}
        card = dashboard._per_event_color_pickers()
        (reset_button,) = card.select(pn.widgets.Button)

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

    def test_save_controls_are_labeled_save_plot(self, dashboard):
        field = dashboard._save_controls(
            options_name="save_options_cont", callback=dashboard.plotter.download_cont, base_name="psth"
        )
        assert field_labels(field) == ["Save Plot"]

    def test_save_controls_expose_format_and_download(self, dashboard):
        row = dashboard._save_controls(
            options_name="save_options_cont", callback=dashboard.plotter.download_cont, base_name="psth"
        )
        assert list(row.select(pn.widgets.Select))
        download = row.select(pn.widgets.FileDownload)
        assert download and download[0].filename == "psth.png"

    def test_save_controls_filename_tracks_format(self, dashboard):
        row = dashboard._save_controls(
            options_name="save_options_cont", callback=dashboard.plotter.download_cont, base_name="psth"
        )
        download = row.select(pn.widgets.FileDownload)[0]
        dashboard.plotter.save_options_cont = "svg"
        assert download.filename == "psth.svg"

    def test_range_number_inputs_param_updates_boxes(self, dashboard):
        row = dashboard._range_number_inputs(name="cont_X")
        minimum, maximum = row[0], row[1]

        dashboard.plotter.cont_X = (1.0, 2.0)

        assert minimum.value == 1.0
        assert maximum.value == 2.0

    def test_range_number_inputs_boxes_update_param(self, dashboard):
        row = dashboard._range_number_inputs(name="cont_Y")
        minimum, maximum = row[0], row[1]

        minimum.value = -3.0
        maximum.value = 4.0

        assert dashboard.plotter.cont_Y == (-3.0, 4.0)

    def test_range_number_inputs_box_edit_moves_live_figure(self, dashboard):
        # End-to-end: typing a box value moves the already-rendered figure in place.
        figure = _FakeFigure()
        dashboard.plotter._figures["cont"] = figure
        row = dashboard._range_number_inputs(name="cont_X")
        minimum, maximum = row[0], row[1]

        minimum.value = 0.0
        maximum.value = 3.0

        assert (figure.x_range.start, figure.x_range.end) == (0.0, 3.0)

    def test_psth_tab_has_hide_minor_ticks_checkbox(self, dashboard, plotter):
        checkboxes = {box.name: box for box in dashboard._psth_tab.select(pn.widgets.Checkbox)}
        assert "Hide Minor Ticks" in checkboxes

        checkbox = checkboxes["Hide Minor Ticks"]
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
        assert "Hide Minor Ticks" in checkboxes

        checkbox = checkboxes["Hide Minor Ticks"]
        assert checkbox.value is False  # ticks shown by default

        checkbox.value = True
        assert plotter.hide_minor_ticks_heatmap is True

    def test_heatmap_tab_has_ticks_inside_checkbox(self, dashboard, plotter):
        checkboxes = {box.name: box for box in dashboard._heatmap_tab.select(pn.widgets.Checkbox)}
        assert "Ticks Inside" in checkboxes

        checkbox = checkboxes["Ticks Inside"]
        assert checkbox.value is False  # ticks point outward by default

        checkbox.value = True
        assert plotter.ticks_inside_heatmap is True

    def test_heatmap_tab_has_hide_outer_border_checkbox(self, dashboard, plotter):
        checkboxes = {box.name: box for box in dashboard._heatmap_tab.select(pn.widgets.Checkbox)}
        assert "Hide Top/Right Border" in checkboxes

        checkbox = checkboxes["Hide Top/Right Border"]
        assert checkbox.value is False  # full border shown by default

        checkbox.value = True
        assert plotter.hide_outer_border_heatmap is True

    def test_heatmap_tab_has_save_controls(self, dashboard):
        (download,) = dashboard._heatmap_tab.select(pn.widgets.FileDownload)
        assert download.filename == "heatmap.png"
        assert "Save Plot" in field_labels(dashboard._heatmap_tab)
