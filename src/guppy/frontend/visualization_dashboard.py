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

        Editing a box writes ``(min, max)`` into ``plotter.<name>``, which
        re-renders the associated plot; the plotter's range-sync hook writing
        into that same param (on a Bokeh zoom/pan) updates the boxes, so the
        numbers always reflect the plot's current axis range.

        Parameters
        ----------
        name : str
            Name of the ``param.Range`` on the plotter (e.g. ``"cont_X"``).
        label : str
            Prefix for the two box labels (e.g. ``"X"`` -> "X min"/"X max").
        """
        plotter = self.plotter
        current = getattr(plotter, name)
        minimum = pn.widgets.FloatInput(
            name=f"{label} min", width=90, value=None if current is None else float(current[0])
        )
        maximum = pn.widgets.FloatInput(
            name=f"{label} max", width=90, value=None if current is None else float(current[1])
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

        plotter.param.watch(from_param, name)
        minimum.param.watch(to_param, "value")
        maximum.param.watch(to_param, "value")
        return pn.Row(minimum, maximum, margin=(0, 15, 0, 0))

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
        psth_y_parameters = pn.Param(
            self.plotter.param.psth_y,
            widgets={
                "psth_y": {
                    "type": pn.widgets.MultiSelect,
                    "name": "Trial # - Timestamps",
                    "width": 200,
                    "size": 15,
                    "align": "start",
                }
            },
        )

        event_selector = pn.Param(
            self.plotter.param.event_selector, widgets={"event_selector": {"type": pn.widgets.Select, "width": 400}}
        )
        x_selector = pn.Param(self.plotter.param.x, widgets={"x": {"type": pn.widgets.Select, "width": 180}})
        y_selector = pn.Param(self.plotter.param.y, widgets={"y": {"type": pn.widgets.Select, "width": 180}})

        width_plot = pn.Param(
            self.plotter.param.Width_Plot, widgets={"Width_Plot": {"type": pn.widgets.Select, "width": 70}}
        )
        height_plot = pn.Param(
            self.plotter.param.Height_Plot, widgets={"Height_Plot": {"type": pn.widgets.Select, "width": 70}}
        )
        ylabel = pn.Param(self.plotter.param.Y_Label, widgets={"Y_Label": {"type": pn.widgets.Select, "width": 70}})
        save_opts = pn.Param(
            self.plotter.param.save_options, widgets={"save_options": {"type": pn.widgets.Select, "width": 70}}
        )

        trace_color = pn.Param(
            self.plotter.param.trace_color, widgets={"trace_color": {"type": pn.widgets.ColorPicker, "width": 90}}
        )
        mean_color = pn.Param(
            self.plotter.param.mean_color, widgets={"mean_color": {"type": pn.widgets.ColorPicker, "width": 90}}
        )
        save_psth = pn.Param(
            self.plotter.param.save_psth, widgets={"save_psth": {"type": pn.widgets.Button, "width": 400}}
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
            pn.Row(width_plot, height_plot, ylabel),
            title="Display settings (all plots)",
            collapsed=False,
        )

        single_event_plot = pn.Card(
            event_selector,
            pn.Row(x_selector, y_selector),
            pn.Row(trace_color, mean_color),
            pn.pane.Markdown("**Axis limits**"),
            cont_limits,
            pn.Row(save_opts, save_psth),
            pn.pane.Markdown("_Save exports this plot and the multi-event comparison plot below._"),
            self.plotter.contPlot,
            title="1 · Single-event PSTH (mean, one trial, or all trials)",
            collapsed=False,
        )

        comparison_plot = pn.Card(
            pn.pane.Markdown("Tick events to overlay their mean traces:"),
            parameters,
            pn.pane.Markdown("**Axis limits**"),
            overlay_limits,
            self.plotter.update_selector,
            title="2 · Multi-event comparison",
            collapsed=False,
        )

        selected_trials_plot = pn.Card(
            pn.pane.Markdown("_Uses the event and trace colors chosen in section 1._"),
            pn.Row(psth_checkbox, psth_y_parameters),
            pn.pane.Markdown("**Axis limits**"),
            trials_limits,
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
        save_hm = pn.Param(self.plotter.param.save_hm, widgets={"save_hm": {"type": pn.widgets.Button, "width": 150}})
        save_options_heatmap = pn.Param(
            self.plotter.param.save_options_heatmap,
            widgets={"save_options_heatmap": {"type": pn.widgets.Select, "width": 150}},
        )

        return pn.Column(
            "## " + self.basename,
            pn.Row(
                event_selector_heatmap,
                color_map,
                width_heatmap,
                height_heatmap,
                save_options_heatmap,
                pn.Column(pn.Spacer(height=25), save_hm),
            ),
            pn.Row(self.plotter.heatmap, heatmap_y_parameters),
        )

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
