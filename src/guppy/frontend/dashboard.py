import logging

import panel as pn

from .frontend_utils import scanPortsAndFind
from .visualization import Viewer

pn.extension()

logger = logging.getLogger(__name__)


class VisualizationDashboard:
    """Dashboard for interactive PSTH and heatmap visualization.

    Composes a ``Viewer`` instance with Panel widgets and a tabbed layout.
    Data loading and preparation are handled externally; this class
    receives already-prepared data and is responsible for widget creation,
    layout assembly, and serving the application.

    Parameters
    ----------
    basename : str
        Session name displayed as the tab title.
    filepath : str
        Output directory path (used by Viewer for saving plots).
    df : pandas.DataFrame
        Concatenated multi-indexed DataFrame of PSTH data.
    columns_dict : dict
        Mapping of event names to their available column names.
    event_options : list
        Event names for the PSTH event selector.
    heatmap_options : list
        Event names for the heatmap event selector.
    multiple_plots_options : list
        Options for the multi-event overlay selector.
    colormaps : list
        Ordered list of matplotlib colormap names.
    x_options : list
        X-axis column options.
    y_options : list
        Y-axis column options.
    trial_options : list
        Trial labels (including "All" as last element).
    x_min : float
        Lower bound for the X-axis range slider.
    x_max : float
        Upper bound for the X-axis range slider.
    """

    def __init__(
        self,
        *,
        basename,
        filepath,
        df,
        columns_dict,
        event_options,
        heatmap_options,
        multiple_plots_options,
        colormaps,
        x_options,
        y_options,
        trial_options,
        x_min,
        x_max,
    ):
        self.basename = basename
        self.viewer = Viewer(
            event_selector_objects=event_options,
            event_selector_heatmap_objects=heatmap_options,
            selector_for_multipe_events_plot_objects=multiple_plots_options,
            columns_dict=columns_dict,
            df_new=df,
            x_min=x_min,
            x_max=x_max,
            color_map_objects=colormaps,
            filepath=filepath,
            x_objects=x_options,
            y_objects=y_options,
            heatmap_y_objects=trial_options,
            psth_y_objects=trial_options[:-1],
        )
        self._psth_tab = self._build_psth_tab()
        self._heatmap_tab = self._build_heatmap_tab()

    def _build_psth_tab(self):
        """Build the PSTH tab with controls and plot panels."""
        view = self.viewer

        psth_checkbox = pn.Param(
            view.param.select_trials_checkbox,
            widgets={
                "select_trials_checkbox": {
                    "type": pn.widgets.CheckBoxGroup,
                    "inline": True,
                    "name": "Select mean and/or just trials",
                }
            },
        )
        parameters = pn.Param(
            view.param.selector_for_multipe_events_plot,
            widgets={
                "selector_for_multipe_events_plot": {"type": pn.widgets.CrossSelector, "width": 550, "align": "start"}
            },
        )
        psth_y_parameters = pn.Param(
            view.param.psth_y,
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
            view.param.event_selector, widgets={"event_selector": {"type": pn.widgets.Select, "width": 400}}
        )
        x_selector = pn.Param(view.param.x, widgets={"x": {"type": pn.widgets.Select, "width": 180}})
        y_selector = pn.Param(view.param.y, widgets={"y": {"type": pn.widgets.Select, "width": 180}})

        width_plot = pn.Param(view.param.Width_Plot, widgets={"Width_Plot": {"type": pn.widgets.Select, "width": 70}})
        height_plot = pn.Param(
            view.param.Height_Plot, widgets={"Height_Plot": {"type": pn.widgets.Select, "width": 70}}
        )
        ylabel = pn.Param(view.param.Y_Label, widgets={"Y_Label": {"type": pn.widgets.Select, "width": 70}})
        save_opts = pn.Param(
            view.param.save_options, widgets={"save_options": {"type": pn.widgets.Select, "width": 70}}
        )

        xlimit_plot = pn.Param(view.param.X_Limit, widgets={"X_Limit": {"type": pn.widgets.RangeSlider, "width": 180}})
        ylimit_plot = pn.Param(view.param.Y_Limit, widgets={"Y_Limit": {"type": pn.widgets.RangeSlider, "width": 180}})
        save_psth = pn.Param(view.param.save_psth, widgets={"save_psth": {"type": pn.widgets.Button, "width": 400}})

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
            view.contPlot,
            view.update_selector,
            view.plot_specific_trials,
        )

    def _build_heatmap_tab(self):
        """Build the heatmap tab with controls and plot panels."""
        view = self.viewer

        heatmap_y_parameters = pn.Param(
            view.param.heatmap_y,
            widgets={
                "heatmap_y": {"type": pn.widgets.MultiSelect, "name": "Trial # - Timestamps", "width": 200, "size": 30}
            },
        )
        event_selector_heatmap = pn.Param(
            view.param.event_selector_heatmap,
            widgets={"event_selector_heatmap": {"type": pn.widgets.Select, "width": 150}},
        )
        color_map = pn.Param(view.param.color_map, widgets={"color_map": {"type": pn.widgets.Select, "width": 150}})
        width_heatmap = pn.Param(
            view.param.width_heatmap, widgets={"width_heatmap": {"type": pn.widgets.Select, "width": 150}}
        )
        height_heatmap = pn.Param(
            view.param.height_heatmap, widgets={"height_heatmap": {"type": pn.widgets.Select, "width": 150}}
        )
        save_hm = pn.Param(view.param.save_hm, widgets={"save_hm": {"type": pn.widgets.Button, "width": 150}})
        save_options_heatmap = pn.Param(
            view.param.save_options_heatmap,
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
            pn.Row(view.heatmap, heatmap_y_parameters),
        )

    def show(self):
        """Serve the dashboard in a browser on an available port."""
        logger.info("app")
        template = pn.template.MaterialTemplate(title="Visualization GUI")
        number = scanPortsAndFind(start_port=5000, end_port=5200)
        app = pn.Tabs(("PSTH", self._psth_tab), ("Heat Map", self._heatmap_tab))
        template.main.append(app)
        template.show(port=number)
