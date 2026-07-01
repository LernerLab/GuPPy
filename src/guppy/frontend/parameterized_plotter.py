import logging
import math
import os
import re
from collections.abc import Callable

import datashader as ds
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
import selenium
from bokeh.io import export_png, export_svgs
from holoviews import opts
from holoviews.operation.datashader import datashade
from holoviews.plotting.util import process_cmap
from selenium.webdriver.chrome.options import Options

pn.extension()

logger = logging.getLogger(__name__)
# Panel registers its bundled JS extensions (es-module-shims, tabulator, luxon) with Bokeh
# using file:// paths. When headless Chrome renders the export HTML it tries to load those
# paths, fails with ERR_FILE_NOT_FOUND, and Bokeh logs a WARNING for each one. The plots
# export correctly regardless, so we suppress these harmless warnings here.
logging.getLogger("bokeh.io.export").setLevel(logging.ERROR)


# remove unnecessary column names
def remove_cols(cols: list[str]) -> list[str]:
    """Remove bookkeeping columns from a PSTH column list.

    Drops ``"err"``, ``"timestamps"``, and any column matching ``bin_err_*``
    so that only trial and mean columns remain.

    Parameters
    ----------
    cols : list of str
        Full list of column names from a PSTH DataFrame.

    Returns
    -------
    list of str
        Filtered column list with bookkeeping columns removed.
    """
    regex = re.compile("bin_err_*")
    remove_cols = [cols[i] for i in range(len(cols)) if regex.match(cols[i])]
    remove_cols = remove_cols + ["err", "timestamps"]
    cols = [i for i in cols if i not in remove_cols]

    return cols


# make a new directory for saving plots
def make_dir(filepath: str) -> str:
    """Create (if needed) and return the ``saved_plots`` subdirectory under ``filepath``.

    Parameters
    ----------
    filepath : str
        Path to the session directory.

    Returns
    -------
    str
        Absolute path to the ``saved_plots`` directory.
    """
    op = os.path.join(filepath, "saved_plots")
    if not os.path.exists(op):
        os.mkdir(op)

    return op


def _headless_chrome_options() -> Options:
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return options


# create a class to make GUI and plot different graphs
class ParameterizedPlotter(param.Parameterized):
    """Interactive Panel/HoloViews dashboard for exploring PSTH results.

    Provides tabbed views for mean PSTH traces, single-trial overlays, trial
    heatmaps, and multi-event comparisons.  All interactive selectors are
    exposed as ``param`` parameters so that Panel can bind them to reactive
    plot callbacks automatically.
    """

    event_selector_objects = param.List(default=None)
    event_selector_heatmap_objects = param.List(default=None)
    selector_for_multipe_events_plot_objects = param.List(default=None)
    color_map_objects = param.List(default=None)
    x_objects = param.List(default=None)
    y_objects = param.List(default=None)
    heatmap_y_objects = param.List(default=None)
    psth_y_objects = param.List(default=None)

    filepath = param.Path(default=None)
    # create different options and selectors
    event_selector = param.ObjectSelector(default=None)
    event_selector_heatmap = param.ObjectSelector(default=None)
    selector_for_multipe_events_plot = param.ListSelector(default=None)
    columns_dict = param.Dict(default=None)
    df_new = param.DataFrame(default=None)
    x_min = param.Number(default=None)
    x_max = param.Number(default=None)
    select_trials_checkbox = param.ListSelector(default=["just trials"], objects=["mean", "just trials"])
    Y_Label = param.ObjectSelector(default="y", objects=["y", "z-score", "\u0394F/F"])
    save_options = param.ObjectSelector(
        default="None", objects=["None", "save_png_format", "save_svg_format", "save_both_format"]
    )
    save_options_heatmap = param.ObjectSelector(
        default="None", objects=["None", "save_png_format", "save_svg_format", "save_both_format"]
    )
    color_map = param.ObjectSelector(default="plasma")
    trace_color = param.Color(default="#0000ff")
    mean_color = param.Color(default="#000000")
    height_heatmap = param.ObjectSelector(default=600, objects=list(np.arange(0, 5100, 100))[1:])
    width_heatmap = param.ObjectSelector(default=1000, objects=list(np.arange(0, 5100, 100))[1:])
    Height_Plot = param.ObjectSelector(default=300, objects=list(np.arange(0, 5100, 100))[1:])
    Width_Plot = param.ObjectSelector(default=1000, objects=list(np.arange(0, 5100, 100))[1:])
    save_hm = param.Action(lambda x: x.param.trigger("save_hm"), label="Save")
    save_psth = param.Action(lambda x: x.param.trigger("save_psth"), label="Save")
    # Each PSTH plot owns an independent pair of axis-range params so that
    # zooming/panning one plot (synced back via _range_sync_hook) does not move
    # the others.  X ranges default to the padded PSTH window (set in __init__);
    # Y ranges start as None and auto-fit to the data on first render.
    cont_X = param.Range(default=None)
    cont_Y = param.Range(default=None)
    overlay_X = param.Range(default=None)
    overlay_Y = param.Range(default=None)
    trials_X = param.Range(default=None)
    trials_Y = param.Range(default=None)

    x = param.ObjectSelector(default=None)
    y = param.ObjectSelector(default=None)
    heatmap_y = param.ListSelector(default=None)
    psth_y = param.ListSelector(default=None)
    results_hm = dict()
    results_psth = dict()

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        # Bind selector objects from companion params
        self.param.event_selector.objects = self.event_selector_objects
        self.param.event_selector_heatmap.objects = self.event_selector_heatmap_objects
        self.param.selector_for_multipe_events_plot.objects = self.selector_for_multipe_events_plot_objects
        self.param.color_map.objects = self.color_map_objects
        self.param.x.objects = self.x_objects
        self.param.y.objects = self.y_objects
        self.param.heatmap_y.objects = self.heatmap_y_objects
        self.param.psth_y.objects = self.psth_y_objects

        # Set defaults
        self.event_selector = self.event_selector_objects[0]
        self.event_selector_heatmap = self.event_selector_heatmap_objects[0]
        self.selector_for_multipe_events_plot = [self.selector_for_multipe_events_plot_objects[0]]
        self.x = self.x_objects[0]
        self.y = self.y_objects[-2]
        self.heatmap_y = [self.heatmap_y_objects[-1]]

        self.cont_X = (self.x_min, self.x_max)
        self.overlay_X = (self.x_min, self.x_max)
        self.trials_X = (self.x_min, self.x_max)

        # Live Bokeh figure per plot, refreshed by _range_sync_hook on each render.
        # Range params are deliberately NOT in the plot methods' @param.depends
        # lists: a range change drives the live figure directly (below) instead of
        # triggering a re-render, so an interactive zoom/pan is never interrupted.
        self._figures: dict[str, object] = {}
        for plot_key, x_name, y_name in self._RANGE_PLOTS:
            self.param.watch(self._make_figure_range_pusher(plot_key, x_name, y_name), [x_name, y_name])

    _RANGE_PLOTS = (
        ("cont", "cont_X", "cont_Y"),
        ("overlay", "overlay_X", "overlay_Y"),
        ("trials", "trials_X", "trials_Y"),
    )

    @staticmethod
    def _assign_range(axis_range: object, bounds: tuple) -> None:
        """Set a Bokeh range's ``start``/``end`` only when they actually differ."""
        low, high = float(bounds[0]), float(bounds[1])
        if (axis_range.start, axis_range.end) != (low, high):
            axis_range.start, axis_range.end = low, high

    def _make_figure_range_pusher(self, plot_key: str, x_name: str, y_name: str) -> Callable[[object], None]:
        """Build a param watcher that pushes a range param onto the live Bokeh figure.

        This is the field/zoom -> plot direction that does not go through a
        re-render: editing a number box (or a programmatic range update) moves the
        already-rendered figure's axes in place.  Writing the same value the figure
        already has is a no-op, so this never fights an in-progress zoom.
        """

        def push(event: object) -> None:
            figure = self._figures.get(plot_key)
            if figure is None:
                return
            x_range, y_range = getattr(self, x_name), getattr(self, y_name)
            if x_range is not None:
                self._assign_range(figure.x_range, x_range)
            if y_range is not None:
                self._assign_range(figure.y_range, y_range)

        return push

    def _range_sync_hook(self, plot_key: str, x_name: str, y_name: str) -> Callable[[object, object], None]:
        """Build a HoloViews ``hooks`` callback that mirrors Bokeh zoom/pan into params.

        Attached to a plot via ``.opts(hooks=[...])``, the returned hook records the
        rendered Bokeh figure (so number-box edits can drive it) and listens for
        changes to the figure's x/y ranges, writing them back into the
        ``x_name``/``y_name`` range params so the plot's number-entry boxes stay in
        sync when the user zooms or pans that plot directly.
        """

        def hook(plot: object, element: object) -> None:
            figure = plot.state
            self._figures[plot_key] = figure

            def sync(attr: str, old: object, new: object) -> None:
                x_range = (figure.x_range.start, figure.x_range.end)
                y_range = (figure.y_range.start, figure.y_range.end)
                if None not in x_range and getattr(self, x_name) != x_range:
                    setattr(self, x_name, (float(x_range[0]), float(x_range[1])))
                if None not in y_range and getattr(self, y_name) != y_range:
                    setattr(self, y_name, (float(y_range[0]), float(y_range[1])))

            for axis_range in (figure.x_range, figure.y_range):
                axis_range.on_change("start", sync)
                axis_range.on_change("end", sync)

        return hook

    # function to save heatmaps when save button on heatmap tab is clicked
    @param.depends("save_hm", watch=True)
    def save_hm_plots(self) -> None:
        """Export the current heatmap to disk in the format selected by ``save_options_heatmap``."""
        plot = self.results_hm["plot"]
        op = self.results_hm["op"]
        save_opts = self.save_options_heatmap
        logger.info(save_opts)
        if save_opts == "None":
            return 0
        with selenium.webdriver.Chrome(options=_headless_chrome_options()) as webdriver:
            if save_opts == "save_svg_format":
                p = hv.render(plot, backend="bokeh")
                p.output_backend = "svg"
                export_svgs(p, filename=op + ".svg", webdriver=webdriver)
            elif save_opts == "save_png_format":
                p = hv.render(plot, backend="bokeh")
                export_png(p, filename=op + ".png", webdriver=webdriver)
            elif save_opts == "save_both_format":
                p = hv.render(plot, backend="bokeh")
                p.output_backend = "svg"
                export_svgs(p, filename=op + ".svg", webdriver=webdriver)
                p_png = hv.render(plot, backend="bokeh")
                export_png(p_png, filename=op + ".png", webdriver=webdriver)

    # function to save PSTH plots when save button on PSTH tab is clicked
    @param.depends("save_psth", watch=True)
    def save_psth_plot(self) -> None:
        """Export the current PSTH plots to disk in the format selected by ``save_options``."""
        plot, op = [], []
        plot.append(self.results_psth["plot_combine"])
        op.append(self.results_psth["op_combine"])
        plot.append(self.results_psth["plot"])
        op.append(self.results_psth["op"])
        save_opts = self.save_options
        if save_opts == "None":
            return 0
        with selenium.webdriver.Chrome(options=_headless_chrome_options()) as webdriver:
            for temp_plot, temp_op in zip(plot, op):
                if save_opts == "save_svg_format":
                    p = hv.render(temp_plot, backend="bokeh")
                    p.output_backend = "svg"
                    export_svgs(p, filename=temp_op + ".svg", webdriver=webdriver)
                elif save_opts == "save_png_format":
                    p = hv.render(temp_plot, backend="bokeh")
                    export_png(p, filename=temp_op + ".png", webdriver=webdriver)
                elif save_opts == "save_both_format":
                    p = hv.render(temp_plot, backend="bokeh")
                    p.output_backend = "svg"
                    export_svgs(p, filename=temp_op + ".svg", webdriver=webdriver)
                    p_png = hv.render(temp_plot, backend="bokeh")
                    export_png(p_png, filename=temp_op + ".png", webdriver=webdriver)

    # function to change Y values based on event selection
    @param.depends("event_selector", watch=True)
    def _update_x_y(self) -> None:
        x_value = self.columns_dict[self.event_selector]
        y_value = self.columns_dict[self.event_selector]
        self.param["x"].objects = [x_value[-4]]
        self.param["y"].objects = remove_cols(y_value)
        self.x = x_value[-4]
        self.y = self.param["y"].objects[-2]

    @param.depends("event_selector_heatmap", watch=True)
    def _update_df(self) -> None:
        cols = self.columns_dict[self.event_selector_heatmap]
        trial_no = range(1, len(remove_cols(cols)[:-2]) + 1)
        trial_ts = ["{} - {}".format(i, j) for i, j in zip(trial_no, remove_cols(cols)[:-2])] + ["All"]
        self.param["heatmap_y"].objects = trial_ts
        self.heatmap_y = [trial_ts[-1]]

    @param.depends("event_selector", watch=True)
    def _update_psth_y(self) -> None:
        cols = self.columns_dict[self.event_selector]
        trial_no = range(1, len(remove_cols(cols)[:-2]) + 1)
        trial_ts = ["{} - {}".format(i, j) for i, j in zip(trial_no, remove_cols(cols)[:-2])]
        self.param["psth_y"].objects = trial_ts
        self.psth_y = [trial_ts[0]]

        # function to plot multiple PSTHs into one plot

    @param.depends(
        "selector_for_multipe_events_plot",
        "Y_Label",
        "save_options",
        "Height_Plot",
        "Width_Plot",
    )
    def update_selector(self) -> hv.NdOverlay | None:
        """Render an overlay of mean PSTH curves for all selected events.

        Returns
        -------
        holoviews.NdOverlay or None
            Overlay of mean curves with spread bands, or ``None`` when no
            events are selected.
        """
        data_curve, cols_curve, data_spread, cols_spread = [], [], [], []
        arr = self.selector_for_multipe_events_plot
        df1 = self.df_new
        for i in range(len(arr)):
            if "bin" in arr[i]:
                split = arr[i].rsplit("_", 2)
                df_name = split[0]  #'{}_{}'.format(split[0], split[1])
                col_name_mean = "{}_{}".format(split[-2], split[-1])
                col_name_err = "{}_err_{}".format(split[-2], split[-1])
                data_curve.append(df1[df_name][col_name_mean])
                cols_curve.append(arr[i])
                data_spread.append(df1[df_name][col_name_err])
                cols_spread.append(arr[i])
            else:
                data_curve.append(df1[arr[i]]["mean"])
                cols_curve.append(arr[i] + "_" + "mean")
                data_spread.append(df1[arr[i]]["err"])
                cols_spread.append(arr[i] + "_" + "mean")

        if len(arr) > 0:
            if self.overlay_Y is None:
                self.overlay_Y = (np.nanmin(np.asarray(data_curve)) - 0.5, np.nanmax(np.asarray(data_curve)) + 0.5)

            if "bin" in arr[i]:
                split = arr[i].rsplit("_", 2)
                df_name = split[0]
                data_curve.append(df1[df_name]["timestamps"])
                cols_curve.append("timestamps")
                data_spread.append(df1[df_name]["timestamps"])
                cols_spread.append("timestamps")
            else:
                data_curve.append(df1[arr[i]]["timestamps"])
                cols_curve.append("timestamps")
                data_spread.append(df1[arr[i]]["timestamps"])
                cols_spread.append("timestamps")
            df_curve = pd.concat(data_curve, axis=1)
            df_spread = pd.concat(data_spread, axis=1)
            df_curve.columns = cols_curve
            df_spread.columns = cols_spread

            ts = df_curve["timestamps"]
            index = np.arange(0, ts.shape[0], 3)
            df_curve = df_curve.loc[index, :]
            df_spread = df_spread.loc[index, :]
            overlay = hv.NdOverlay(
                {
                    c: hv.Curve((df_curve["timestamps"], df_curve[c]), kdims=["Time (s)"]).opts(
                        width=int(self.Width_Plot),
                        height=int(self.Height_Plot),
                        xlim=self.overlay_X,
                        ylim=self.overlay_Y,
                    )
                    for c in cols_curve[:-1]
                }
            )
            spread = hv.NdOverlay(
                {
                    d: hv.Spread(
                        (df_spread["timestamps"], df_curve[d], df_spread[d], df_spread[d]),
                        vdims=["y", "yerrpos", "yerrneg"],
                    ).opts(line_width=0, fill_alpha=0.3)
                    for d in cols_spread[:-1]
                }
            )
            plot_combine = ((overlay * spread).opts(opts.NdOverlay(xlabel="Time (s)", ylabel=self.Y_Label))).opts(
                shared_axes=False
            )
            # plot_err = new_df.hvplot.area(x='timestamps', y=[], y2=[])
            save_opts = self.save_options
            op = make_dir(self.filepath)
            op_filename = os.path.join(op, str(arr) + "_mean")

            plot_combine = plot_combine.opts(hooks=[self._range_sync_hook("overlay", "overlay_X", "overlay_Y")])
            self.results_psth["plot_combine"] = plot_combine
            self.results_psth["op_combine"] = op_filename
            # self.save_plots(plot_combine, save_opts, op_filename)
            return plot_combine

    # function to plot mean PSTH, single trial in PSTH and all the trials of PSTH with mean
    @param.depends(
        "event_selector",
        "x",
        "y",
        "Y_Label",
        "save_options",
        "Height_Plot",
        "Width_Plot",
        "trace_color",
        "mean_color",
    )
    def contPlot(self) -> hv.Element:
        """Render the selected PSTH view (mean, single trial, or all-trials datashaded overlay).

        Returns
        -------
        holoviews.Element
            A ``Curve``, ``Spread``-overlaid ``Curve``, or datashaded
            ``NdOverlay``, depending on the value of ``y``.
        """
        df1 = self.df_new[self.event_selector]
        # height = self.Heigth_Plot
        # width = self.Width_Plot
        # logger.info(height, width)
        if self.y == "All":
            if self.cont_Y is None:
                self.cont_Y = (np.nanmin(np.asarray(df1)) - 0.5, np.nanmax(np.asarray(df1)) - 0.5)

            options = self.param["y"].objects
            regex = re.compile("bin_[(]")
            remove_bin_trials = [options[i] for i in range(len(options)) if not regex.match(options[i])]

            ndoverlay = hv.NdOverlay({c: hv.Curve((df1[self.x], df1[c])) for c in remove_bin_trials[:-2]})
            img1 = datashade(ndoverlay, normalization="linear", aggregator=ds.count())
            x_points = df1[self.x]
            y_points = df1["mean"]
            img2 = hv.Curve((x_points, y_points))
            img = (img1 * img2).opts(
                opts.Curve(
                    width=int(self.Width_Plot),
                    height=int(self.Height_Plot),
                    line_width=4,
                    color=self.mean_color,
                    xlim=self.cont_X,
                    ylim=self.cont_Y,
                    xlabel="Time (s)",
                    ylabel=self.Y_Label,
                )
            )

            img = img.opts(hooks=[self._range_sync_hook("cont", "cont_X", "cont_Y")])
            save_opts = self.save_options

            op = make_dir(self.filepath)
            op_filename = os.path.join(op, self.event_selector + "_" + self.y)
            self.results_psth["plot"] = img
            self.results_psth["op"] = op_filename
            # self.save_plots(img, save_opts, op_filename)

            return img

        elif self.y == "mean" or "bin" in self.y:

            xpoints = df1[self.x]
            ypoints = df1[self.y]
            if self.y == "mean":
                err = df1["err"]
            else:
                split = self.y.split("_")
                err = df1["{}_err_{}".format(split[0], split[1])]

            index = np.arange(0, xpoints.shape[0], 3)

            if self.cont_Y is None:
                self.cont_Y = (np.nanmin(ypoints) - 0.5, np.nanmax(ypoints) + 0.5)

            ropts_curve = dict(
                width=int(self.Width_Plot),
                height=int(self.Height_Plot),
                xlim=self.cont_X,
                ylim=self.cont_Y,
                color=self.trace_color,
                xlabel="Time (s)",
                ylabel=self.Y_Label,
            )
            ropts_spread = dict(
                width=int(self.Width_Plot),
                height=int(self.Height_Plot),
                fill_alpha=0.3,
                fill_color=self.trace_color,
                line_width=0,
            )

            plot_curve = hv.Curve((xpoints[index], ypoints[index]))  # .opts(**ropts_curve)
            plot_spread = hv.Spread(
                (xpoints[index], ypoints[index], err[index], err[index])
            )  # .opts(**ropts_spread) #vdims=['y', 'yerrpos', 'yerrneg']
            plot = (plot_curve * plot_spread).opts({"Curve": ropts_curve, "Spread": ropts_spread})
            plot = plot.opts(hooks=[self._range_sync_hook("cont", "cont_X", "cont_Y")])

            save_opts = self.save_options
            op = make_dir(self.filepath)
            op_filename = os.path.join(op, self.event_selector + "_" + self.y)
            self.results_psth["plot"] = plot
            self.results_psth["op"] = op_filename
            # self.save_plots(plot, save_opts, op_filename)

            return plot

        else:
            xpoints = df1[self.x]
            ypoints = df1[self.y]
            if self.cont_Y is None:
                self.cont_Y = (np.nanmin(ypoints) - 0.5, np.nanmax(ypoints) + 0.5)

            ropts_curve = dict(
                width=int(self.Width_Plot),
                height=int(self.Height_Plot),
                xlim=self.cont_X,
                ylim=self.cont_Y,
                color=self.trace_color,
                xlabel="Time (s)",
                ylabel=self.Y_Label,
            )
            plot = hv.Curve((xpoints, ypoints)).opts({"Curve": ropts_curve})
            plot = plot.opts(hooks=[self._range_sync_hook("cont", "cont_X", "cont_Y")])

            save_opts = self.save_options
            op = make_dir(self.filepath)
            op_filename = os.path.join(op, self.event_selector + "_" + self.y)
            self.results_psth["plot"] = plot
            self.results_psth["op"] = op_filename
            # self.save_plots(plot, save_opts, op_filename)

            return plot

    # function to plot specific PSTH trials
    @param.depends(
        "event_selector",
        "x",
        "psth_y",
        "select_trials_checkbox",
        "Y_Label",
        "save_options",
        "Height_Plot",
        "Width_Plot",
        "trace_color",
        "mean_color",
    )
    def plot_specific_trials(self) -> hv.Element | None:
        """Render the user-selected subset of PSTH trials, optionally with their mean.

        Returns
        -------
        holoviews.Element or None
            An overlay of individual trial curves, a mean-with-spread curve, or
            their combination, depending on ``select_trials_checkbox``.  Returns
            ``None`` when ``psth_y`` is not set.
        """
        df_psth = self.df_new[self.event_selector]
        # trials_Y is left as None here so the y-axis auto-fits per selection.

        if self.psth_y == None:
            return None
        else:
            selected_trials = [s.split(" - ")[1] for s in list(self.psth_y)]

        if self.trials_Y is None:
            selected = np.asarray(df_psth[selected_trials])
            self.trials_Y = (np.nanmin(selected) - 0.5, np.nanmax(selected) + 0.5)

        index = np.arange(0, df_psth["timestamps"].shape[0], 3)

        if self.select_trials_checkbox == ["just trials"]:
            overlay = hv.NdOverlay(
                {
                    c: hv.Curve((df_psth["timestamps"][index], df_psth[c][index]), kdims=["Time (s)"])
                    for c in selected_trials
                }
            )
            ropts = dict(
                width=int(self.Width_Plot),
                height=int(self.Height_Plot),
                xlim=self.trials_X,
                ylim=self.trials_Y,
                xlabel="Time (s)",
                ylabel=self.Y_Label,
                hooks=[self._range_sync_hook("trials", "trials_X", "trials_Y")],
            )
            return overlay.opts(**ropts)
        elif self.select_trials_checkbox == ["mean"]:
            arr = np.asarray(df_psth[selected_trials])
            mean = np.nanmean(arr, axis=1)
            err = np.nanstd(arr, axis=1) / math.sqrt(arr.shape[1])
            ropts_curve = dict(
                width=int(self.Width_Plot),
                height=int(self.Height_Plot),
                xlim=self.trials_X,
                ylim=self.trials_Y,
                color=self.trace_color,
                xlabel="Time (s)",
                ylabel=self.Y_Label,
            )
            ropts_spread = dict(
                width=int(self.Width_Plot),
                height=int(self.Height_Plot),
                fill_alpha=0.3,
                fill_color=self.trace_color,
                line_width=0,
            )
            plot_curve = hv.Curve((df_psth["timestamps"][index], mean[index]))
            plot_spread = hv.Spread((df_psth["timestamps"][index], mean[index], err[index], err[index]))
            plot = (plot_curve * plot_spread).opts({"Curve": ropts_curve, "Spread": ropts_spread})
            plot = plot.opts(hooks=[self._range_sync_hook("trials", "trials_X", "trials_Y")])
            return plot
        elif self.select_trials_checkbox == ["mean", "just trials"]:
            overlay = hv.NdOverlay(
                {
                    c: hv.Curve((df_psth["timestamps"][index], df_psth[c][index]), kdims=["Time (s)"])
                    for c in selected_trials
                }
            )
            ropts_overlay = dict(
                width=int(self.Width_Plot),
                height=int(self.Height_Plot),
                xlim=self.trials_X,
                ylim=self.trials_Y,
                xlabel="Time (s)",
                ylabel=self.Y_Label,
            )

            arr = np.asarray(df_psth[selected_trials])
            mean = np.nanmean(arr, axis=1)
            err = np.nanstd(arr, axis=1) / math.sqrt(arr.shape[1])
            ropts_curve = dict(
                width=int(self.Width_Plot),
                height=int(self.Height_Plot),
                xlim=self.trials_X,
                ylim=self.trials_Y,
                color=self.mean_color,
                xlabel="Time (s)",
                ylabel=self.Y_Label,
            )
            ropts_spread = dict(
                width=int(self.Width_Plot),
                height=int(self.Height_Plot),
                fill_alpha=0.3,
                fill_color=self.mean_color,
                line_width=0,
            )
            plot_curve = hv.Curve((df_psth["timestamps"][index], mean[index]))
            plot_spread = hv.Spread((df_psth["timestamps"][index], mean[index], err[index], err[index]))

            plot = (plot_curve * plot_spread).opts({"Curve": ropts_curve, "Spread": ropts_spread})
            combined = overlay.opts(**ropts_overlay) * plot
            return combined.opts(hooks=[self._range_sync_hook("trials", "trials_X", "trials_Y")])

    # function to show heatmaps for each event
    @param.depends("event_selector_heatmap", "color_map", "height_heatmap", "width_heatmap", "heatmap_y")
    def heatmap(self) -> hv.Element:
        """Render a trial heatmap for the selected event.

        Returns
        -------
        holoviews.Element
            A ``QuadMesh`` (single trial) or a datashaded ``QuadMesh`` overlay
            (multiple trials), coloured by the selected colour map.
        """
        height = self.height_heatmap
        width = self.width_heatmap
        df_hm = self.df_new[self.event_selector_heatmap]
        cols = list(df_hm.columns)
        regex = re.compile("bin_err_*")
        drop_cols = [cols[i] for i in range(len(cols)) if regex.match(cols[i])]
        drop_cols = ["err", "mean"] + drop_cols
        df_hm = df_hm.drop(drop_cols, axis=1)
        cols = list(df_hm.columns)
        bin_cols = [cols[i] for i in range(len(cols)) if re.compile("bin_*").match(cols[i])]
        time = np.asarray(df_hm["timestamps"])
        event_ts_for_each_event = np.arange(1, len(df_hm.columns[:-1]) + 1)
        yticks = list(event_ts_for_each_event)
        z_score = np.asarray(df_hm[df_hm.columns[:-1]]).T

        if self.heatmap_y[0] == "All":
            indices = np.arange(z_score.shape[0] - len(bin_cols))
            z_score = z_score[indices, :]
            event_ts_for_each_event = np.arange(1, z_score.shape[0] + 1)
            yticks = list(event_ts_for_each_event)
        else:
            remove_all = list(set(self.heatmap_y) - set(["All"]))
            indices = sorted([int(s.split("-")[0]) - 1 for s in remove_all])
            z_score = z_score[indices, :]
            event_ts_for_each_event = np.arange(1, z_score.shape[0] + 1)
            yticks = list(event_ts_for_each_event)

        clim = (np.nanmin(z_score), np.nanmax(z_score))
        font_size = {"labels": 16, "yticks": 6}

        if event_ts_for_each_event.shape[0] == 1:
            dummy_image = hv.QuadMesh((time, event_ts_for_each_event, z_score)).opts(colorbar=True, clim=clim)
            image = (
                (dummy_image).opts(
                    opts.QuadMesh(
                        width=int(width),
                        height=int(height),
                        cmap=process_cmap(self.color_map, provider="matplotlib"),
                        colorbar=True,
                        ylabel="Trials",
                        xlabel="Time (s)",
                        fontsize=font_size,
                        yticks=yticks,
                    )
                )
            ).opts(shared_axes=False)

            save_opts = self.save_options_heatmap
            op = make_dir(self.filepath)
            op_filename = os.path.join(op, self.event_selector_heatmap + "_" + "heatmap")
            self.results_hm["plot"] = image
            self.results_hm["op"] = op_filename
            # self.save_plots(image, save_opts, op_filename)
            return image
        else:
            ropts = dict(
                width=int(width),
                height=int(height),
                ylabel="Trials",
                xlabel="Time (s)",
                fontsize=font_size,
                yticks=yticks,
                invert_yaxis=True,
            )
            dummy_image = hv.QuadMesh((time[0:100], event_ts_for_each_event, z_score[:, 0:100])).opts(
                colorbar=True, cmap=process_cmap(self.color_map, provider="matplotlib"), clim=clim
            )
            actual_image = hv.QuadMesh((time, event_ts_for_each_event, z_score))

            dynspread_img = datashade(actual_image, cmap=process_cmap(self.color_map, provider="matplotlib")).opts(
                **ropts
            )  # clims=self.C_Limit, cnorm='log'
            image = ((dummy_image * dynspread_img).opts(opts.QuadMesh(width=int(width), height=int(height)))).opts(
                shared_axes=False
            )

            save_opts = self.save_options_heatmap
            op = make_dir(self.filepath)
            op_filename = os.path.join(op, self.event_selector_heatmap + "_" + "heatmap")
            self.results_hm["plot"] = image
            self.results_hm["op"] = op_filename

            return image
