from collections.abc import Callable
from io import BytesIO

import panel as pn

from .binned_metrics_view import build_binned_metrics_view
from .covariate_correlation_view import build_covariate_correlation_view
from .parameterized_plotter import METRIC_LABELS, build_plotter
from .psth_significance_view import build_psth_significance_view
from .tonic_epochs import build_tonic_results_view

pn.extension()


def _controls(*controls: object) -> pn.FlexBox:
    """Lay a card's controls out in one row that wraps onto as many lines as the page width needs.

    Parameters
    ----------
    *controls : object
        Panel objects to lay out, in reading order.

    Returns
    -------
    panel.FlexBox
        The controls, stretched to the width of the card.
    """
    return pn.FlexBox(*controls, sizing_mode="stretch_width")


def _labeled(*, label: str, content: object) -> pn.Column:
    """Keep a control group under its bold label, so the two wrap together.

    Parameters
    ----------
    label : str
        Text shown above the group.
    content : object
        The Panel object the label describes.

    Returns
    -------
    panel.Column
        The label above its content.
    """
    return pn.Column(pn.pane.Markdown(f"**{label}**", margin=(5, 10, 0, 10)), content)


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


class VisualizationDashboard:
    """Dashboard for interactive PSTH and heatmap visualization.

    Wraps a ``Viewer`` instance with Panel widgets and a tabbed layout.
    Data loading, preparation, and Viewer instantiation are handled
    externally; this class is responsible for widget creation, layout
    assembly, and serving the application.

    Parameters
    ----------
    plotter : ParameterizedPlotter
        A fully configured ParameterizedPlotter instance that provides reactive plot
        methods and param-based controls.
    basename : str
        Session name displayed as the tab title.
    events : list of str
        Event labels the plotter was built over, used to rebuild it for another metric.
    metric : str
        The metric ``plotter`` holds results for: ``"z_score"`` or ``"dff"``.
    available_metrics : list of str
        Metrics step 4 wrote complete results for in this output directory. The options
        of the metric selector the PSTH and Heat Map tabs each carry.
    """

    def __init__(
        self,
        *,
        plotter: object,
        basename: str,
        events: list[str],
        metric: str,
        available_metrics: list[str],
    ) -> None:
        self.plotter = plotter
        self.basename = basename
        self.events = events
        self.metric = metric
        self.available_metrics = available_metrics
        self._psth_tab = self._build_psth_tab()
        self._heatmap_tab = self._build_heatmap_tab()
        self._tonic_tab = build_tonic_results_view(plotter.filepath, label=basename)
        self._binned_tab = build_binned_metrics_view(plotter.filepath)
        self._covariate_tab = build_covariate_correlation_view(plotter.filepath)
        self._significance_tab = build_psth_significance_view(plotter.filepath)
        self._tabs = pn.Tabs(
            ("PSTH", self._psth_tab),
            ("Heat Map", self._heatmap_tab),
            ("Tonic", self._tonic_tab),
            ("Binned", self._binned_tab),
            ("Covariates", self._covariate_tab),
            ("Significance", self._significance_tab),
            sizing_mode="stretch_width",
        )

    def _metric_control(self) -> pn.widgets.Select:
        """Return one tab's metric chooser, placed among the controls it drives.

        The PSTH and Heat Map tabs each get their own instance. Both tabs are rebuilt on
        every change, so the two stay in step without being linked.
        """
        select = pn.widgets.Select(
            name="Metric",
            options={METRIC_LABELS[name]: name for name in self.available_metrics},
            value=self.metric,
            width=140,
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
            filepath=self.plotter.filepath,
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

    def _range_number_inputs(self, *, name: str, label: str) -> pn.Row:
        """Return two-way-bound min/max number boxes for a plotter ``Range`` param.

        Editing a box writes ``(min, max)`` into ``plotter.<name>`` and moves the
        live figure in place via ``move_figure_to_range``; conversely the plotter's
        range-sync hook writing into that same param (on a Bokeh zoom/pan) updates
        the boxes, so the numbers always reflect the plot's current axis range.

        Parameters
        ----------
        name : str
            Name of the ``param.Range`` on the plotter (e.g. ``"cont_X"``).
        label : str
            Prefix for the two box labels (e.g. ``"X"`` -> "X min"/"X max").
        """
        plotter = self.plotter
        current = getattr(plotter, name)
        # format="0.00" keeps the displayed value to two decimals (the underlying
        # float value is unrounded); width=110 leaves room for a couple of digits.
        minimum = pn.widgets.FloatInput(
            name=f"{label} min", width=110, format="0.00", value=None if current is None else float(current[0])
        )
        maximum = pn.widgets.FloatInput(
            name=f"{label} max", width=110, format="0.00", value=None if current is None else float(current[1])
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
        return pn.Row(minimum, maximum, margin=(0, 15, 0, 0))

    def _save_controls(self, *, options_name: str, callback: Callable[[], BytesIO], base_name: str) -> pn.Row:
        """Return a format selector + a "Save As…" download button for one plot.

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
        options = pn.Param(
            getattr(self.plotter.param, options_name),
            widgets={options_name: {"type": pn.widgets.Select, "width": 90, "name": "Format"}},
        )
        download = pn.widgets.FileDownload(
            callback=callback,
            filename=f"{base_name}.{getattr(self.plotter, options_name)}",
            label="Save As…",
            button_type="default",
            width=140,
        )

        def _sync_extension(event: object) -> None:
            download.filename = f"{base_name}.{event.new}"

        self.plotter.param.watch(_sync_extension, options_name)
        return pn.Row(options, pn.Column(pn.Spacer(height=25), download))

    def _per_event_color_pickers(self) -> pn.Card:
        """Return a collapsed card of per-event color pickers for the comparison plot.

        Holds one ``ColorPicker`` per currently-selected event (each seeded with the
        event's effective color -- its override, else its palette color) inside a
        fixed-height scrollable column, plus a "Reset to Palette" button that clears
        all overrides. The picker list is rebuilt whenever the event selection or the
        palette changes so it always matches the events being plotted.
        """
        plotter = self.plotter
        # Roomy enough for long event names to sit on one line so only the vertical
        # scrollbar ever appears; the swatch itself is kept small and square (below)
        # and the name is a separate label, so the swatch is not stretched to fit.
        pickers = pn.Column(scroll="y", height=250, width=520)

        def rebuild(*events: object) -> None:
            effective_colors = plotter.overlay_effective_colors()
            rows = []
            for event, color in effective_colors.items():
                picker = pn.widgets.ColorPicker(value=color, width=50)

                def to_override(picker_event: object, event: str = event) -> None:
                    plotter.overlay_color_overrides = {**plotter.overlay_color_overrides, event: picker_event.new}

                picker.param.watch(to_override, "value")
                label = pn.pane.Markdown(event, margin=(8, 0, 0, 6))
                rows.append(pn.Row(picker, label))
            pickers.objects = rows

        reset = pn.widgets.Button(name="Reset to Palette", width=150)

        def on_reset(event: object) -> None:
            plotter.overlay_color_overrides = {}
            rebuild()

        reset.on_click(on_reset)
        plotter.param.watch(rebuild, ["selector_for_multipe_events_plot", "overlay_palette"])
        rebuild()

        return pn.Card(pickers, reset, title="Per-event colors (optional)", collapsed=True)

    def _build_psth_tab(self) -> pn.Column:
        """Build the PSTH tab with controls and plot panels."""
        psth_checkbox = pn.Param(
            self.plotter.param.select_trials_checkbox,
            widgets={
                "select_trials_checkbox": {
                    "type": pn.widgets.CheckBoxGroup,
                    "inline": True,
                    "name": "Select Mean and/or Just Trials",
                }
            },
        )
        parameters = pn.Param(
            self.plotter.param.selector_for_multipe_events_plot,
            widgets={
                "selector_for_multipe_events_plot": {"type": pn.widgets.CrossSelector, "width": 550, "align": "start"}
            },
        )
        # A CrossSelector (like the multi-event picker) makes the chosen trials
        # visually obvious in a dedicated "selected" column.
        psth_y_parameters = pn.Param(
            self.plotter.param.psth_y,
            widgets={
                "psth_y": {"type": pn.widgets.CrossSelector, "name": "Trials to Show", "width": 420, "align": "start"}
            },
        )

        event_selector = pn.Param(
            self.plotter.param.event_selector, widgets={"event_selector": {"type": pn.widgets.Select, "width": 400}}
        )
        # The x-axis is always "timestamps" (single option), so no x-selector is shown.
        y_selector = pn.Param(
            self.plotter.param.y, widgets={"y": {"type": pn.widgets.Select, "name": "View", "width": 180}}
        )

        width_plot = pn.Param(
            self.plotter.param.Width_Plot,
            widgets={"Width_Plot": {"type": pn.widgets.Select, "name": "Saved Width", "width": 90}},
        )
        height_plot = pn.Param(
            self.plotter.param.Height_Plot,
            widgets={"Height_Plot": {"type": pn.widgets.Select, "name": "Height", "width": 90}},
        )
        hide_minor_ticks = pn.Param(
            self.plotter.param.hide_minor_ticks,
            widgets={"hide_minor_ticks": {"type": pn.widgets.Checkbox, "name": "Hide minor tick marks"}},
        )

        trace_color = pn.Param(
            self.plotter.param.trace_color, widgets={"trace_color": {"type": pn.widgets.ColorPicker, "width": 90}}
        )
        mean_color = pn.Param(
            self.plotter.param.mean_color, widgets={"mean_color": {"type": pn.widgets.ColorPicker, "width": 90}}
        )
        overlay_palette = pn.Param(
            self.plotter.param.overlay_palette,
            widgets={"overlay_palette": {"type": pn.widgets.Select, "width": 150}},
        )
        per_event_colors = self._per_event_color_pickers()

        # Independent save controls (format selector + button) for each plot.
        save_cont = self._save_controls(
            options_name="save_options_cont", callback=self.plotter.download_cont, base_name="psth"
        )
        save_overlay = self._save_controls(
            options_name="save_options_overlay", callback=self.plotter.download_overlay, base_name="psth_comparison"
        )
        save_trials = self._save_controls(
            options_name="save_options_trials", callback=self.plotter.download_trials, base_name="psth_trials"
        )

        # Each plot owns an independent axis-limit control that also snaps to Bokeh zoom/pan.
        cont_limits = pn.Row(
            self._range_number_inputs(name="cont_X", label="X"), self._range_number_inputs(name="cont_Y", label="Y")
        )
        overlay_limits = pn.Row(
            self._range_number_inputs(name="overlay_X", label="X"),
            self._range_number_inputs(name="overlay_Y", label="Y"),
        )
        trials_limits = pn.Row(
            self._range_number_inputs(name="trials_X", label="X"),
            self._range_number_inputs(name="trials_Y", label="Y"),
        )

        # Controls that affect every plot live in one shared card; each plot then
        # gets its own card holding only the controls that drive it, so it is
        # always clear which control belongs to which plot.
        shared_settings = pn.Card(
            _controls(self._metric_control(), width_plot, height_plot, hide_minor_ticks),
            title="Display Settings (all plots)",
            collapsed=False,
            sizing_mode="stretch_width",
        )

        single_event_plot = pn.Card(
            _controls(
                event_selector,
                y_selector,
                trace_color,
                _labeled(label="Axis Limits", content=cont_limits),
                save_cont,
            ),
            _plot(self.plotter.contPlot),
            title="1 · Single-Event PSTH (mean, one trial, or all trials)",
            collapsed=False,
            sizing_mode="stretch_width",
        )

        comparison_plot = pn.Card(
            _controls(
                _labeled(label="Tick events to overlay their mean traces:", content=parameters),
                overlay_palette,
                per_event_colors,
                _labeled(label="Axis Limits", content=overlay_limits),
                save_overlay,
            ),
            _plot(self.plotter.update_selector),
            title="2 · Multi-Event Comparison",
            collapsed=False,
            sizing_mode="stretch_width",
        )

        selected_trials_plot = pn.Card(
            pn.pane.Markdown(
                "_Uses the event and trace color chosen in section 1. Mean color (below) colors the mean overlay._"
            ),
            _controls(
                psth_checkbox,
                mean_color,
                _labeled(label="Move trials into the right-hand column to plot them:", content=psth_y_parameters),
                _labeled(label="Axis Limits", content=trials_limits),
                save_trials,
            ),
            _plot(self.plotter.plot_specific_trials),
            title="3 · Selected Trials (of the section-1 event)",
            collapsed=False,
            sizing_mode="stretch_width",
        )

        return pn.Column(
            "## " + self.basename,
            shared_settings,
            single_event_plot,
            comparison_plot,
            selected_trials_plot,
            sizing_mode="stretch_width",
        )

    def _build_heatmap_tab(self) -> pn.Column:
        """Build the heatmap tab with controls and plot panels."""
        heatmap_y_parameters = pn.Param(
            self.plotter.param.heatmap_y,
            widgets={
                "heatmap_y": {"type": pn.widgets.MultiSelect, "name": "Trial # - Timestamps", "width": 200, "size": 30}
            },
        )
        event_selector_heatmap = pn.Param(
            self.plotter.param.event_selector_heatmap,
            widgets={"event_selector_heatmap": {"type": pn.widgets.Select, "width": 150}},
        )
        color_map = pn.Param(
            self.plotter.param.color_map, widgets={"color_map": {"type": pn.widgets.Select, "width": 150}}
        )
        width_heatmap = pn.Param(
            self.plotter.param.width_heatmap,
            widgets={"width_heatmap": {"type": pn.widgets.Select, "name": "Saved Width", "width": 150}},
        )
        height_heatmap = pn.Param(
            self.plotter.param.height_heatmap,
            widgets={"height_heatmap": {"type": pn.widgets.Select, "name": "Height", "width": 150}},
        )
        hide_minor_ticks_heatmap = pn.Param(
            self.plotter.param.hide_minor_ticks_heatmap,
            widgets={"hide_minor_ticks_heatmap": {"type": pn.widgets.Checkbox, "name": "Hide minor tick marks"}},
        )
        # Opt-in styling toggles (defaults keep Bokeh's standard look).
        ticks_inside_heatmap = pn.Param(
            self.plotter.param.ticks_inside_heatmap,
            widgets={"ticks_inside_heatmap": {"type": pn.widgets.Checkbox, "name": "Ticks inside"}},
        )
        hide_outer_border_heatmap = pn.Param(
            self.plotter.param.hide_outer_border_heatmap,
            widgets={"hide_outer_border_heatmap": {"type": pn.widgets.Checkbox, "name": "Hide top/right border"}},
        )

        # Independent save controls (format selector + button), matching the PSTH plots.
        save_hm = self._save_controls(
            options_name="save_options_heatmap", callback=self.plotter.download_heatmap, base_name="heatmap"
        )

        # Numeric X-axis limits (snap to Bokeh zoom/pan) and the colour-scale (clim)
        # limits, both reusing the PSTH tab's _range_number_inputs helper. The X boxes
        # move the live figure in place; the colour-scale boxes re-render (heatmap_clim
        # is in heatmap()'s @param.depends), for which move_figure_to_range is a no-op.
        # The Trials (Y) axis is intentionally not exposed: it always spans every trial
        # row at full height, computed fresh on each render.
        axis_limits = self._range_number_inputs(name="heatmap_X", label="X")
        color_limits = self._range_number_inputs(name="heatmap_clim", label="Color Scale")

        heatmap_card = pn.Card(
            _controls(
                event_selector_heatmap,
                self._metric_control(),
                color_map,
                width_heatmap,
                height_heatmap,
                hide_minor_ticks_heatmap,
                ticks_inside_heatmap,
                hide_outer_border_heatmap,
                _labeled(label="Axis Limits", content=axis_limits),
                _labeled(label="Color Scale Limits", content=color_limits),
                save_hm,
            ),
            pn.Row(_plot(self.plotter.heatmap), heatmap_y_parameters, sizing_mode="stretch_width"),
            title="Trial heatmap",
            collapsed=False,
            sizing_mode="stretch_width",
        )

        return pn.Column("## " + self.basename, heatmap_card, sizing_mode="stretch_width")

    def build_template(self) -> pn.template.MaterialTemplate:
        """Build and return the Panel template without serving it."""
        template = pn.template.MaterialTemplate(title="Visualization GUI")
        template.main.append(self._tabs)
        return template
