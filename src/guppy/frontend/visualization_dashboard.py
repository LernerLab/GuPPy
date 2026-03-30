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

    def __init__(self, *, plotter, basename):
        self.plotter = plotter
        self.basename = basename
        self._psth_tab = self._build_psth_tab()
        self._heatmap_tab = self._build_heatmap_tab()

    def _build_psth_tab(self):
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

        xlimit_plot = pn.Param(
            self.plotter.param.X_Limit, widgets={"X_Limit": {"type": pn.widgets.RangeSlider, "width": 180}}
        )
        ylimit_plot = pn.Param(
            self.plotter.param.Y_Limit, widgets={"Y_Limit": {"type": pn.widgets.RangeSlider, "width": 180}}
        )
        save_psth = pn.Param(
            self.plotter.param.save_psth, widgets={"save_psth": {"type": pn.widgets.Button, "width": 400}}
        )

        options = pn.Column(
            event_selector,
            pn.Row(x_selector, y_selector),
            pn.Row(xlimit_plot, ylimit_plot),
            pn.Row(width_plot, height_plot, ylabel, save_opts),
            save_psth,
        )

        options_selectors = pn.Row(options, parameters)

        return pn.Column(
            "## " + self.basename,
            pn.Row(options_selectors, pn.Column(psth_checkbox, psth_y_parameters), width=1200),
            self.plotter.contPlot,
            self.plotter.update_selector,
            self.plotter.plot_specific_trials,
        )

    def _build_heatmap_tab(self):
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

    def build_template(self):
        """Build and return the Panel template without serving it."""
        template = pn.template.MaterialTemplate(title="Visualization GUI")
        app = pn.Tabs(("PSTH", self._psth_tab), ("Heat Map", self._heatmap_tab))
        template.main.append(app)
        return template

    def show(self):
        """Serve the dashboard in a browser on an available port."""
        logger.info("app")
        template = self.build_template()
        number = scanPortsAndFind(start_port=5000, end_port=5200)
        template.show(port=number)
