from collections.abc import Callable
from io import BytesIO

import panel as pn

from .binned_metrics_view import binned_metrics_sites, build_binned_metrics_view
from .covariate_correlation_view import (
    build_covariate_correlation_view,
    covariate_correlation_sites,
)
from .parameterized_plotter import METRIC_LABELS, build_plotter
from .plot_card import (
    CHECKBOX_MARGIN,
    CONTROL_MARGIN,
    plot_card,
    plot_with_side_panel,
    range_separator,
    toolbar_field,
)
from .psth_significance_view import build_psth_significance_view
from .tonic_epochs import _tonic_result_sites, build_tonic_results_view

pn.extension()


def _plot(plot_method: Callable[[], object]) -> pn.viewable.Viewable:
    """Show a plotter method's plot across the full width of its card.

    Parameters
    ----------
    plot_method : callable
        A plotter method decorated with ``param.depends``, re-run whenever its parameters change.

    Returns
    -------
    panel.viewable.Viewable
        The reactive pane, stretched to the width of its container.
    """
    return pn.panel(plot_method, sizing_mode="stretch_width")


def whole_session_result_tabs(*, filepath: str) -> list[str]:
    """Return the names of the event-independent tabs that have results in ``filepath``.

    Parameters
    ----------
    filepath : str
        Output directory: a session run folder or a group folder.

    Returns
    -------
    list of str
        A subset of ``["Tonic", "Binned", "Covariates"]``, in tab order.
    """
    sites_by_tab = {
        "Tonic": _tonic_result_sites(filepath),
        "Binned": binned_metrics_sites(filepath),
        "Covariates": covariate_correlation_sites(filepath),
    }
    return [tab for tab, sites in sites_by_tab.items() if sites]


class VisualizationDashboard:
    """Dashboard for interactive PSTH and heatmap visualization.

    Wraps a ``Viewer`` instance with Panel widgets and a tabbed layout.
    Data loading, preparation, and Viewer instantiation are handled
    externally; this class is responsible for widget creation, layout
    assembly, and serving the application.

    With a plotter, the dashboard shows the PSTH, Heat Map, Tonic, Binned, Covariates
    and Significance tabs. Without one, the output directory has no PSTH results, and
    the dashboard shows only the Tonic, Binned and Covariates tabs that have results.

    Parameters
    ----------
    filepath : str
        Output directory the dashboard reads its results from.
    plotter : ParameterizedPlotter or None
        A fully configured ParameterizedPlotter instance that provides reactive plot
        methods and param-based controls, or None when the output directory holds no
        PSTH results.
    basename : str
        Session name displayed as the tab title.
    events : list of str
        Event labels the plotter was built over, used to rebuild it for another metric.
    metric : str or None
        The metric ``plotter`` holds results for: ``"z_score"`` or ``"dff"``. None
        without a plotter.
    available_metrics : list of str
        Metrics step 4 wrote complete results for in this output directory. The options
        of the metric selector the PSTH and Heat Map tabs each carry.
    """

    def __init__(
        self,
        *,
        filepath: str,
        plotter: object | None,
        basename: str,
        events: list[str],
        metric: str | None,
        available_metrics: list[str],
    ) -> None:
        self.filepath = filepath
        self.plotter = plotter
        self.basename = basename
        self.events = events
        self.metric = metric
        self.available_metrics = available_metrics
        self._tonic_tab = build_tonic_results_view(filepath)
        self._binned_tab = build_binned_metrics_view(filepath)
        self._covariate_tab = build_covariate_correlation_view(filepath)
        if plotter is None:
            tabs_by_name = {"Tonic": self._tonic_tab, "Binned": self._binned_tab, "Covariates": self._covariate_tab}
            self._tabs = pn.Tabs(
                *[(name, tabs_by_name[name]) for name in whole_session_result_tabs(filepath=filepath)],
                dynamic=True,
                sizing_mode="stretch_width",
            )
            return

        self._psth_tab = self._build_psth_tab()
        self._heatmap_tab = self._build_heatmap_tab()
        self._significance_tab = build_psth_significance_view(filepath)
        self._tabs = pn.Tabs(
            ("PSTH", self._psth_tab),
            ("Heat Map", self._heatmap_tab),
            ("Tonic", self._tonic_tab),
            ("Binned", self._binned_tab),
            ("Covariates", self._covariate_tab),
            ("Significance", self._significance_tab),
            # A tab's plots are drawn when it is first opened, so a multi-panel layout sizes
            # itself against the visible page; drawn while hidden, it collapses to zero width.
            dynamic=True,
            sizing_mode="stretch_width",
        )

    def _metric_control(self) -> pn.widgets.Select:
        """Return one tab's metric chooser, placed among the controls it drives.

        The PSTH and Heat Map tabs each get their own instance. Both tabs are rebuilt on
        every change, so the two stay in step without being linked.
        """
        select = pn.widgets.Select(
            name="",
            options={METRIC_LABELS[name]: name for name in self.available_metrics},
            value=self.metric,
            width=140,
            margin=CONTROL_MARGIN,
        )
        select.param.watch(self._on_metric_change, "value")
        return select

    def _on_metric_change(self, event: object) -> None:
        """Reload the PSTH and Heat Map tabs against the newly selected metric.

        The plotter sets every selector's option list in its constructor and has no
        reload path, so a metric switch builds a new one. The remaining four tabs read
        their own files from the output directory and are left untouched.
        """
        self._tabs.loading = True
        self.metric = event.new
        self.plotter = build_plotter(
            filepath=self.filepath,
            events=self.events,
            metric=event.new,
            x_min=self.plotter.x_min,
            x_max=self.plotter.x_max,
        )
        self._psth_tab = self._build_psth_tab()
        self._heatmap_tab = self._build_heatmap_tab()
        self._tabs[0] = ("PSTH", self._psth_tab)
        self._tabs[1] = ("Heat Map", self._heatmap_tab)
        self._tabs.loading = False

    def _range_number_inputs(self, *, name: str) -> pn.Row:
        """Return two-way-bound low/high number boxes for a plotter ``Range`` param.

        Editing a box writes ``(low, high)`` into ``plotter.<name>`` and moves the
        live figure in place via ``move_figure_to_range``; conversely the plotter's
        range-sync hook writing into that same param (on a Bokeh zoom/pan) updates
        the boxes, so the numbers always reflect the plot's current axis range.

        Parameters
        ----------
        name : str
            Name of the ``param.Range`` on the plotter (e.g. ``"cont_X"``).

        Returns
        -------
        panel.Row
            The low box, then the high box.
        """
        plotter = self.plotter
        current = getattr(plotter, name)
        # format="0.00" keeps the displayed value to two decimals (the underlying
        # float value is unrounded).
        minimum = pn.widgets.FloatInput(
            width=88, format="0.00", margin=CONTROL_MARGIN, value=None if current is None else float(current[0])
        )
        maximum = pn.widgets.FloatInput(
            width=88, format="0.00", margin=CONTROL_MARGIN, value=None if current is None else float(current[1])
        )
        # Guard so param->box updates don't echo back as box->param writes.
        state = {"syncing": False}

        def from_param(event: object) -> None:
            state["syncing"] = True
            value = event.new
            minimum.value = None if value is None else float(value[0])
            maximum.value = None if value is None else float(value[1])
            state["syncing"] = False

        def to_param(event: object) -> None:
            if state["syncing"] or minimum.value is None or maximum.value is None:
                return
            new_range = (minimum.value, maximum.value)
            if getattr(plotter, name) != new_range:
                setattr(plotter, name, new_range)
                # A typed box edit is the one case that must move the already-rendered
                # figure in place; a render writes these params too, but must not.
                plotter.move_figure_to_range(name)

        plotter.param.watch(from_param, name)
        minimum.param.watch(to_param, "value")
        maximum.param.watch(to_param, "value")
        return pn.Row(minimum, maximum, margin=0)

    def _range_field(self, *, name: str, label: str) -> pn.Column:
        """Return a toolbar field holding a plotter ``Range`` as two boxes joined by a dash.

        Parameters
        ----------
        name : str
            Name of the ``param.Range`` on the plotter (e.g. ``"cont_X"``).
        label : str
            The field's label (e.g. "X Range").
        """
        minimum, maximum = self._range_number_inputs(name=name)
        return toolbar_field(label, minimum, range_separator(), maximum)

    def _save_controls(self, *, options_name: str, callback: Callable[[], BytesIO], base_name: str) -> pn.Column:
        """Return the "Save Plot" toolbar field: a format selector and a "Save As…" download button.

        The button streams the freshly rendered plot to the browser, which prompts
        the user for a filename and location (the standard download flow). The
        suggested filename tracks the chosen format's extension.

        Parameters
        ----------
        options_name : str
            Name of the plotter's ``png``/``svg`` format param for this plot.
        callback : callable
            Zero-argument function returning the encoded image buffer to download.
        base_name : str
            Default filename stem shown in the browser's save dialog.
        """
        save_format = pn.widgets.Select.from_param(
            getattr(self.plotter.param, options_name), name="", width=80, margin=CONTROL_MARGIN
        )
        download = pn.widgets.FileDownload(
            callback=callback,
            filename=f"{base_name}.{getattr(self.plotter, options_name)}",
            label="Save As…",
            button_type="default",
            width=110,
            margin=CONTROL_MARGIN,
        )

        def _sync_extension(event: object) -> None:
            download.filename = f"{base_name}.{event.new}"

        self.plotter.param.watch(_sync_extension, options_name)
        return toolbar_field("Save Plot", save_format, download)

    def _per_event_color_pickers(self) -> pn.FlexBox:
        """Return the strip of per-event color pickers for the comparison plot, hidden until opened.

        Holds one ``ColorPicker`` per currently-selected event (each seeded with the
        event's effective color -- its override, else its palette color), plus a
        "Reset to Palette" button that clears all overrides. The pickers are rebuilt
        whenever the event selection or the palette changes so they always match the
        events being plotted.
        """
        plotter = self.plotter
        pickers = pn.FlexBox(margin=0)

        def rebuild(*events: object) -> None:
            effective_colors = plotter.overlay_effective_colors()
            rows = []
            for event, color in effective_colors.items():
                picker = pn.widgets.ColorPicker(value=color, width=50, margin=CONTROL_MARGIN)

                def to_override(picker_event: object, event: str = event) -> None:
                    plotter.overlay_color_overrides = {**plotter.overlay_color_overrides, event: picker_event.new}

                picker.param.watch(to_override, "value")
                label = pn.pane.Markdown(event, margin=(0, 16, 0, 0), align="center")
                rows.append(pn.Row(picker, label, margin=(0, 0, 4, 0)))
            pickers.objects = rows

        reset = pn.widgets.Button(name="Reset to Palette", button_type="light", width=150, margin=CONTROL_MARGIN)

        def on_reset(event: object) -> None:
            plotter.overlay_color_overrides = {}
            rebuild()

        reset.on_click(on_reset)
        plotter.param.watch(rebuild, ["selector_for_multipe_events_plot", "overlay_palette"])
        rebuild()

        return pn.FlexBox(
            pickers,
            reset,
            visible=False,
            sizing_mode="stretch_width",
            styles={"border-bottom": "1px solid #E9ECEF", "padding-bottom": "8px"},
            margin=(0, 10, 8, 10),
        )

    def _cross_selector(self, parameter: object, *, height: object) -> pn.widgets.CrossSelector:
        """Return a CrossSelector for a side panel, as narrow as 300px and stretching to the panel's width.

        A CrossSelector sizes its two lists from its width once, when built, and keeps that size
        as their minimum; built at 300px and then stretched, the lists grow and shrink with the panel.
        """
        picker = pn.widgets.CrossSelector.from_param(
            parameter, name="", width=300, height=height, margin=CONTROL_MARGIN
        )
        picker.sizing_mode = "stretch_width"
        return picker

    def _follow_plot_height(self, height_parameter: object, *, offset: int) -> object:
        """Bind a height to a plot height parameter, less ``offset`` pixels (e.g. for a label above)."""
        return pn.bind(lambda height: int(height) - offset, height_parameter)

    def _build_psth_tab(self) -> pn.Column:
        """Build the PSTH tab: shared display settings, then one card per plot."""
        plotter = self.plotter

        hide_minor_ticks = pn.widgets.Checkbox.from_param(
            plotter.param.hide_minor_ticks, name="Hide Minor Ticks", margin=CHECKBOX_MARGIN
        )
        display_settings = plot_card(
            title="Display Settings",
            note="These apply to every plot on this tab.",
            fields=[
                toolbar_field("Metric", self._metric_control()),
                toolbar_field(
                    "Saved Width",
                    pn.widgets.Select.from_param(plotter.param.Width_Plot, name="", width=90, margin=CONTROL_MARGIN),
                ),
                toolbar_field(
                    "Height",
                    pn.widgets.Select.from_param(plotter.param.Height_Plot, name="", width=90, margin=CONTROL_MARGIN),
                ),
                toolbar_field("Style", hide_minor_ticks),
            ],
        )

        # The x-axis is always "timestamps" (single option), so no x-selector is shown.
        single_event_plot = plot_card(
            title="Single-Event PSTH",
            note="The mean, one trial, or all trials of one event.",
            fields=[
                toolbar_field(
                    "Event",
                    pn.widgets.Select.from_param(
                        plotter.param.event_selector, name="", width=260, margin=CONTROL_MARGIN
                    ),
                ),
                toolbar_field(
                    "View", pn.widgets.Select.from_param(plotter.param.y, name="", width=160, margin=CONTROL_MARGIN)
                ),
                toolbar_field(
                    "Trace Color",
                    pn.widgets.ColorPicker.from_param(
                        plotter.param.trace_color, name="", width=56, margin=CONTROL_MARGIN
                    ),
                ),
                self._range_field(name="cont_X", label="X Range"),
                self._range_field(name="cont_Y", label="Y Range"),
            ],
            trailing_field=self._save_controls(
                options_name="save_options_cont", callback=plotter.download_cont, base_name="psth"
            ),
            body=[_plot(plotter.contPlot)],
        )

        # The pickers sit in side panels beside their plots, as tall as the plot, so each toolbar
        # stays one line; each panel can be dragged wider or hidden.
        plot_height = self._follow_plot_height(plotter.param.Height_Plot, offset=0)
        picker_height = self._follow_plot_height(plotter.param.Height_Plot, offset=30)
        event_picker = self._cross_selector(plotter.param.selector_for_multipe_events_plot, height=picker_height)
        per_event_colors = self._per_event_color_pickers()
        colors_toggle = pn.widgets.Toggle(name="Colors", button_type="light", width=90, margin=CONTROL_MARGIN)
        colors_toggle.link(per_event_colors, value="visible")
        comparison_plot = plot_card(
            title="Multi-Event Comparison",
            note="The mean trace of each chosen event, overlaid.",
            fields=[
                toolbar_field(
                    "Palette",
                    pn.widgets.Select.from_param(
                        plotter.param.overlay_palette, name="", width=130, margin=CONTROL_MARGIN
                    ),
                    colors_toggle,
                ),
                self._range_field(name="overlay_X", label="X Range"),
                self._range_field(name="overlay_Y", label="Y Range"),
            ],
            trailing_field=self._save_controls(
                options_name="save_options_overlay", callback=plotter.download_overlay, base_name="psth_comparison"
            ),
            body=[
                per_event_colors,
                plot_with_side_panel(
                    _plot(plotter.update_selector), label="Events", picker=event_picker, height=plot_height
                ),
            ],
        )

        trial_picker = self._cross_selector(plotter.param.psth_y, height=picker_height)
        selected_trials_plot = plot_card(
            title="Selected Trials",
            note="Trials of the Single-Event PSTH's event, in its trace color.",
            fields=[
                toolbar_field(
                    "Show",
                    pn.widgets.CheckBoxGroup.from_param(
                        plotter.param.select_trials_checkbox, name="", inline=True, margin=CHECKBOX_MARGIN
                    ),
                ),
                toolbar_field(
                    "Mean Color",
                    pn.widgets.ColorPicker.from_param(
                        plotter.param.mean_color, name="", width=56, margin=CONTROL_MARGIN
                    ),
                ),
                self._range_field(name="trials_X", label="X Range"),
                self._range_field(name="trials_Y", label="Y Range"),
            ],
            trailing_field=self._save_controls(
                options_name="save_options_trials", callback=plotter.download_trials, base_name="psth_trials"
            ),
            body=[
                plot_with_side_panel(
                    _plot(plotter.plot_specific_trials), label="Trials", picker=trial_picker, height=plot_height
                )
            ],
        )

        return pn.Column(
            display_settings,
            single_event_plot,
            comparison_plot,
            selected_trials_plot,
            sizing_mode="stretch_width",
        )

    def _build_heatmap_tab(self) -> pn.Column:
        """Build the Heat Map tab: its display settings, then the heat map card."""
        plotter = self.plotter

        style_checkboxes = [
            pn.widgets.Checkbox.from_param(getattr(plotter.param, parameter), name=name, margin=CHECKBOX_MARGIN)
            for parameter, name in (
                ("hide_minor_ticks_heatmap", "Hide Minor Ticks"),
                ("ticks_inside_heatmap", "Ticks Inside"),
                ("hide_outer_border_heatmap", "Hide Top/Right Border"),
            )
        ]
        display_settings = plot_card(
            title="Display Settings",
            note="These apply to the heat map on this tab.",
            fields=[
                toolbar_field("Metric", self._metric_control()),
                toolbar_field(
                    "Saved Width",
                    pn.widgets.Select.from_param(plotter.param.width_heatmap, name="", width=90, margin=CONTROL_MARGIN),
                ),
                toolbar_field(
                    "Height",
                    pn.widgets.Select.from_param(
                        plotter.param.height_heatmap, name="", width=90, margin=CONTROL_MARGIN
                    ),
                ),
                toolbar_field("Style", *style_checkboxes),
            ],
        )

        # The Trials (Y) axis is intentionally not exposed: it always spans every trial
        # row at full height, computed fresh on each render. The colour-scale boxes
        # re-render (heatmap_clim is in heatmap()'s @param.depends), for which
        # move_figure_to_range is a no-op.
        trial_picker = pn.widgets.MultiSelect.from_param(
            plotter.param.heatmap_y,
            name="",
            sizing_mode="stretch_width",
            height=self._follow_plot_height(plotter.param.height_heatmap, offset=30),
            margin=CONTROL_MARGIN,
        )
        heatmap_card = plot_card(
            title="Trial Heat Map",
            note="One row per trial of the chosen event.",
            fields=[
                toolbar_field(
                    "Event",
                    pn.widgets.Select.from_param(
                        plotter.param.event_selector_heatmap, name="", width=260, margin=CONTROL_MARGIN
                    ),
                ),
                toolbar_field(
                    "Color Map",
                    pn.widgets.Select.from_param(plotter.param.color_map, name="", width=130, margin=CONTROL_MARGIN),
                ),
                self._range_field(name="heatmap_X", label="X Range"),
                self._range_field(name="heatmap_clim", label="Color Scale"),
            ],
            trailing_field=self._save_controls(
                options_name="save_options_heatmap", callback=plotter.download_heatmap, base_name="heatmap"
            ),
            body=[
                plot_with_side_panel(
                    _plot(plotter.heatmap),
                    label="Trials",
                    picker=trial_picker,
                    height=self._follow_plot_height(plotter.param.height_heatmap, offset=0),
                    side_width=300,
                )
            ],
        )

        return pn.Column(display_settings, heatmap_card, sizing_mode="stretch_width")

    def build_template(self) -> pn.template.MaterialTemplate:
        """Build and return the Panel template without serving it."""
        template = pn.template.MaterialTemplate(title="Visualization GUI")
        template.main.append(self._tabs)
        return template
