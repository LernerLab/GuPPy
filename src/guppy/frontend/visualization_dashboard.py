import logging

import panel as pn

from .frontend_utils import scanPortsAndFind

pn.extension()

logger = logging.getLogger(__name__)


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
    """

    def __init__(self, *, plotter: object, basename: str) -> None:
        self.plotter = plotter
        self.basename = basename
        self._psth_tab = self._build_psth_tab()
        self._heatmap_tab = self._build_heatmap_tab()

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

    def _save_controls(self, *, options_name: str, action_name: str) -> pn.Row:
        """Return a format selector + Save button bound to one plot's save params."""
        options = pn.Param(
            getattr(self.plotter.param, options_name),
            widgets={options_name: {"type": pn.widgets.Select, "width": 150, "name": "Save format"}},
        )
        button = pn.Param(
            getattr(self.plotter.param, action_name),
            widgets={action_name: {"type": pn.widgets.Button, "width": 120}},
        )
        return pn.Row(options, pn.Column(pn.Spacer(height=25), button))

    def _per_event_color_pickers(self) -> pn.Card:
        """Return a collapsed card of per-event color pickers for the comparison plot.

        Holds one ``ColorPicker`` per currently-selected event (each seeded with the
        event's effective color -- its override, else its palette color) inside a
        fixed-height scrollable column, plus a "Reset to palette" button that clears
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

        reset = pn.widgets.Button(name="Reset to palette", width=150)

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
                    "name": "Select mean and/or just trials",
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
                "psth_y": {"type": pn.widgets.CrossSelector, "name": "Trials to show", "width": 420, "align": "start"}
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
            self.plotter.param.Width_Plot, widgets={"Width_Plot": {"type": pn.widgets.Select, "width": 70}}
        )
        height_plot = pn.Param(
            self.plotter.param.Height_Plot, widgets={"Height_Plot": {"type": pn.widgets.Select, "width": 70}}
        )
        ylabel = pn.Param(self.plotter.param.Y_Label, widgets={"Y_Label": {"type": pn.widgets.Select, "width": 70}})
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
        save_cont = self._save_controls(options_name="save_options_cont", action_name="save_cont")
        save_overlay = self._save_controls(options_name="save_options_overlay", action_name="save_overlay")
        save_trials = self._save_controls(options_name="save_options_trials", action_name="save_trials")

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
            pn.Row(width_plot, height_plot, ylabel),
            hide_minor_ticks,
            title="Display settings (all plots)",
            collapsed=False,
        )

        single_event_plot = pn.Card(
            event_selector,
            y_selector,
            pn.Row(trace_color),
            pn.pane.Markdown("**Axis limits**"),
            cont_limits,
            save_cont,
            self.plotter.contPlot,
            title="1 · Single-event PSTH (mean, one trial, or all trials)",
            collapsed=False,
        )

        comparison_plot = pn.Card(
            pn.pane.Markdown("Tick events to overlay their mean traces:"),
            parameters,
            pn.Row(overlay_palette),
            per_event_colors,
            pn.pane.Markdown("**Axis limits**"),
            overlay_limits,
            save_overlay,
            self.plotter.update_selector,
            title="2 · Multi-event comparison",
            collapsed=False,
        )

        selected_trials_plot = pn.Card(
            pn.pane.Markdown(
                "_Uses the event and trace color chosen in section 1. Mean color (below) colors the mean overlay._"
            ),
            psth_checkbox,
            pn.Row(mean_color),
            pn.pane.Markdown("Move trials into the right-hand column to plot them:"),
            psth_y_parameters,
            pn.pane.Markdown("**Axis limits**"),
            trials_limits,
            save_trials,
            self.plotter.plot_specific_trials,
            title="3 · Selected trials (of the section-1 event)",
            collapsed=False,
        )

        return pn.Column(
            "## " + self.basename,
            shared_settings,
            single_event_plot,
            comparison_plot,
            selected_trials_plot,
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
            self.plotter.param.width_heatmap, widgets={"width_heatmap": {"type": pn.widgets.Select, "width": 150}}
        )
        height_heatmap = pn.Param(
            self.plotter.param.height_heatmap, widgets={"height_heatmap": {"type": pn.widgets.Select, "width": 150}}
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
        save_hm = self._save_controls(options_name="save_options_heatmap", action_name="save_hm")

        # Numeric X-axis limits (snap to Bokeh zoom/pan) and the colour-scale (clim)
        # limits, both reusing the PSTH tab's _range_number_inputs helper. The X boxes
        # move the live figure in place; the colour-scale boxes re-render (heatmap_clim
        # is in heatmap()'s @param.depends), for which move_figure_to_range is a no-op.
        # The Trials (Y) axis is intentionally not exposed: it always spans every trial
        # row at full height, computed fresh on each render.
        axis_limits = self._range_number_inputs(name="heatmap_X", label="X")
        color_limits = self._range_number_inputs(name="heatmap_clim", label="Color scale")

        heatmap_card = pn.Card(
            pn.Row(event_selector_heatmap, color_map, width_heatmap, height_heatmap),
            pn.Row(hide_minor_ticks_heatmap, ticks_inside_heatmap, hide_outer_border_heatmap),
            pn.pane.Markdown("**Axis limits**"),
            axis_limits,
            pn.pane.Markdown("**Color scale limits**"),
            color_limits,
            save_hm,
            pn.Row(self.plotter.heatmap, heatmap_y_parameters),
            title="Trial heatmap",
            collapsed=False,
        )

        return pn.Column("## " + self.basename, heatmap_card)

    def build_template(self) -> pn.template.MaterialTemplate:
        """Build and return the Panel template without serving it."""
        template = pn.template.MaterialTemplate(title="Visualization GUI")
        app = pn.Tabs(("PSTH", self._psth_tab), ("Heat Map", self._heatmap_tab))
        template.main.append(app)
        return template

    def show(self) -> None:
        """Serve the dashboard in a browser on an available port."""
        logger.info("app")
        template = self.build_template()
        number = scanPortsAndFind(start_port=5000, end_port=5200)
        template.show(port=number)
