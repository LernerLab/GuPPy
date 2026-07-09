import logging
import math
import os
import re
from collections.abc import Callable
from io import BytesIO

import datashader as ds
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
import selenium
from bokeh.io.export import get_screenshot_as_png, get_svgs
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


def overview_y_options(cols: list[str]) -> list[str]:
    """Whole-event y-selector options for the single-event overview plot.

    Filters :func:`remove_cols` output down to the whole-event views — ``"mean"``,
    the datashaded ``"All"`` trials, and any ``bin_*`` subset averages — dropping
    individual trial columns, which are inspected in the selected-trials plot.

    Parameters
    ----------
    cols : list of str
        Full list of column names from a PSTH DataFrame (including ``"All"``).

    Returns
    -------
    list of str
        Options containing only ``"mean"``, ``"All"``, and bin averages.
    """
    bin_pattern = re.compile("bin_")
    return [col for col in remove_cols(cols) if col in ("mean", "All") or bin_pattern.match(col)]


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


def _headless_chrome_options() -> Options:  # pragma: no cover - configures a real browser subprocess
    options = Options()
    # Modern Chrome (109+) needs the "=new" form; the bare "--headless" flag is
    # deprecated and can pop a visible blank window instead of running headless.
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
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
    _SAVE_FORMATS = ["png", "svg"]
    # Independent save-format selector per plot so each can be exported on its own.
    save_options_cont = param.ObjectSelector(default="png", objects=_SAVE_FORMATS)
    save_options_overlay = param.ObjectSelector(default="png", objects=_SAVE_FORMATS)
    save_options_trials = param.ObjectSelector(default="png", objects=_SAVE_FORMATS)
    save_options_heatmap = param.ObjectSelector(default="png", objects=_SAVE_FORMATS)
    color_map = param.ObjectSelector(default="plasma")
    trace_color = param.Color(default="#0000ff")
    mean_color = param.Color(default="#000000")
    # When True, hide the minor tick marks between axis numbers on the PSTH line plots
    # for a cleaner look. Ticks are shown by default; the user opts in to hide them.
    hide_minor_ticks = param.Boolean(default=False)
    # Categorical palette used to colour the per-event curves in the comparison plot.
    overlay_palette = param.ObjectSelector(
        default="Category10", objects=["Category10", "Category20", "Colorblind", "Dark2", "Set1"]
    )
    # Per-event colour overrides for the comparison plot, keyed by the selection label
    # (the entries of ``selector_for_multipe_events_plot``). Only events the user has
    # explicitly recoloured appear here; every other event falls back to ``overlay_palette``.
    overlay_color_overrides = param.Dict(default={})
    height_heatmap = param.ObjectSelector(default=600, objects=list(np.arange(0, 5100, 100))[1:])
    width_heatmap = param.ObjectSelector(default=1000, objects=list(np.arange(0, 5100, 100))[1:])
    Height_Plot = param.ObjectSelector(default=300, objects=list(np.arange(0, 5100, 100))[1:])
    Width_Plot = param.ObjectSelector(default=1000, objects=list(np.arange(0, 5100, 100))[1:])
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
    # Heat map axis + color-scale ranges, mirroring the per-line-plot pattern.
    # heatmap_X is driven live by the zoom/pan hook (kept OUT of @param.depends);
    # heatmap_clim IS in @param.depends because a colour-scale change has no live-pan
    # equivalent and must re-render. Both auto-fit on selection change. The Trials
    # (Y) axis has no user control: it always shows every trial row at full height,
    # so its ylim is computed fresh on each render rather than stored as a param.
    heatmap_X = param.Range(default=None)
    heatmap_clim = param.Range(default=None)
    # Independent of the PSTH hide_minor_ticks toggle.
    hide_minor_ticks_heatmap = param.Boolean(default=False)
    # Opt-in heatmap styling toggles (defaults keep Bokeh's standard look): draw the
    # axis ticks pointing inward, and drop the top/right frame lines for an open look.
    ticks_inside_heatmap = param.Boolean(default=False)
    hide_outer_border_heatmap = param.Boolean(default=False)

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
        # _update_x_y (fired by setting event_selector above) has already filtered the
        # y options to the overview views; take the default from those filtered options.
        self.y = self.param["y"].objects[-2]
        self.heatmap_y = [self.heatmap_y_objects[-1]]

        self.cont_X = (self.x_min, self.x_max)
        self.overlay_X = (self.x_min, self.x_max)
        self.trials_X = (self.x_min, self.x_max)
        self.heatmap_X = (self.x_min, self.x_max)

        # Live Bokeh figure per plot, recorded by _range_sync_hook on each render.
        # Range params are deliberately NOT in the plot methods' @param.depends
        # lists: a typed-box edit drives the live figure directly (via
        # move_figure_to_range, called from the box callback) instead of triggering
        # a re-render, so an interactive zoom/pan is never interrupted.
        self._figures: dict[str, object] = {}

        # Track the data selection each plot last rendered so the Y range can
        # auto-refit when the selection (event, trace, trials) changes, while
        # still honouring a manual zoom/typed range within a single selection.
        self._selection_keys: dict[str, object] = {}

        # Set while rendering a plot for a "Save As…" download. The off-screen
        # export figure must not be mistaken for the live one, so _range_sync_hook
        # skips recording it (see _render_download).
        self._exporting = False

    _RANGE_PLOTS = (
        ("cont", "cont_X", "cont_Y"),
        ("overlay", "overlay_X", "overlay_Y"),
        ("trials", "trials_X", "trials_Y"),
        ("heatmap", "heatmap_X", None),
    )

    @staticmethod
    def _assign_range(axis_range: object, bounds: tuple) -> None:
        """Set a Bokeh range's ``start``/``end`` only when they actually differ."""
        low, high = float(bounds[0]), float(bounds[1])
        if (axis_range.start, axis_range.end) != (low, high):
            axis_range.start, axis_range.end = low, high

    def move_figure_to_range(self, name: str) -> None:
        """Move a plot's live Bokeh figure to match its current range params.

        Called from the number-box callback so that typing a new limit moves the
        already-rendered figure in place (no re-render). Writing the same value the
        figure already has is a no-op, so this never fights an in-progress zoom.

        This is deliberately NOT a watcher on the range params: a render also writes
        those params (to auto-fit and to keep the boxes in sync), and mutating the
        stale figure mid-re-render is illegal under Bokeh's document lock. Only a
        genuine box edit needs to move the figure, so only the box edit calls this.

        Parameters
        ----------
        name : str
            Name of the range param that changed (e.g. ``"cont_X"``); the matching
            plot's figure is moved to its current x and y ranges.
        """
        for plot_key, x_name, y_name in self._RANGE_PLOTS:
            if name in (x_name, y_name):
                break
        else:
            return
        figure = self._figures.get(plot_key)
        if figure is None:
            return
        x_range = getattr(self, x_name)
        y_range = None if y_name is None else getattr(self, y_name)
        if x_range is not None:
            self._assign_range(figure.x_range, x_range)
        if y_range is not None:
            self._assign_range(figure.y_range, y_range)

    def _range_sync_hook(
        self, plot_key: str, x_name: str, y_name: str | None = None
    ) -> Callable[[object, object], None]:
        """Build a HoloViews ``hooks`` callback that mirrors Bokeh zoom/pan into params.

        Attached to a plot via ``.opts(hooks=[...])``, the returned hook records the
        rendered Bokeh figure (so number-box edits can drive it) and listens for
        changes to the figure's x/y ranges, writing them back into the
        ``x_name``/``y_name`` range params so the plot's number-entry boxes stay in
        sync when the user zooms or pans that plot directly.

        Parameters
        ----------
        plot_key : str
            Key under which the rendered figure is recorded in ``self._figures``.
        x_name : str
            Name of the ``param.Range`` mirrored from the figure's x range.
        y_name : str or None, optional
            Name of the ``param.Range`` mirrored from the figure's y range, or
            ``None`` to leave the y range unsynced (the heatmap's Trials axis is
            fixed, so its range must never be written back).
        """

        def hook(plot: object, element: object) -> None:
            # A "Save As…" export renders an off-screen throwaway figure; recording it
            # (or wiring its range listeners) would leave the live controls driving a
            # dead figure until a browser refresh, so skip the hook entirely then.
            if self._exporting:
                return
            figure = plot.state
            self._figures[plot_key] = figure

            def sync(attr: str, old: object, new: object) -> None:
                x_range = (figure.x_range.start, figure.x_range.end)
                if None not in x_range and getattr(self, x_name) != x_range:
                    setattr(self, x_name, (float(x_range[0]), float(x_range[1])))
                if y_name is not None:
                    y_range = (figure.y_range.start, figure.y_range.end)
                    if None not in y_range and getattr(self, y_name) != y_range:
                        setattr(self, y_name, (float(y_range[0]), float(y_range[1])))

            axis_ranges = (figure.x_range,) if y_name is None else (figure.x_range, figure.y_range)
            for axis_range in axis_ranges:
                axis_range.on_change("start", sync)
                axis_range.on_change("end", sync)

        return hook

    def _hide_minor_ticks_hook(self, attr: str) -> Callable[[object, object], None]:
        """Build a HoloViews ``hooks`` callback that hides the Bokeh axis minor ticks.

        Attached to a plot via ``.opts(hooks=[...])``, the returned hook removes the
        small tick marks between the axis numbers on both axes when the boolean param
        named ``attr`` is set. When it is not set the freshly rendered figure keeps
        Bokeh's default minor ticks, so no restore step is needed. The param name is a
        factory argument so the PSTH plots and the heatmap can each drive their own
        independent toggle (``hide_minor_ticks`` vs ``hide_minor_ticks_heatmap``).
        """

        def hook(plot: object, element: object) -> None:
            if not getattr(self, attr):
                return
            figure = plot.state
            figure.xaxis.minor_tick_line_color = None
            figure.yaxis.minor_tick_line_color = None

        return hook

    def _ticks_inside_hook(self, attr: str) -> Callable[[object, object], None]:
        """Build a HoloViews ``hooks`` callback that points the axis ticks inward.

        Attached to a plot via ``.opts(hooks=[...])``, the returned hook flips both
        axes' major and minor ticks to point into the plot area when the boolean
        param named ``attr`` is set. When it is not set the freshly rendered figure
        keeps Bokeh's default outward ticks, so no restore step is needed.
        """

        def hook(plot: object, element: object) -> None:
            if not getattr(self, attr):
                return
            figure = plot.state
            for axis in (figure.xaxis, figure.yaxis):
                axis.major_tick_in, axis.major_tick_out = 6, 0
                axis.minor_tick_in, axis.minor_tick_out = 4, 0

        return hook

    def _hide_outer_border_hook(self, attr: str) -> Callable[[object, object], None]:
        """Build a HoloViews ``hooks`` callback that removes the top/right frame lines.

        Attached to a plot via ``.opts(hooks=[...])``, the returned hook drops the
        figure's four-sided outline when the boolean param named ``attr`` is set. The
        bottom (x) and left (y) axis lines are drawn separately by Bokeh and remain,
        so only the top and right edges disappear, leaving an open L-shaped frame.
        When the param is not set the freshly rendered figure keeps its full outline,
        so no restore step is needed.
        """

        def hook(plot: object, element: object) -> None:
            if not getattr(self, attr):
                return
            plot.state.outline_line_color = None

        return hook

    def _reset_y_on_selection_change(self, plot_key: str, y_name: str, selection: object) -> None:
        """Clear a plot's Y range when its data selection changes, forcing a re-autofit.

        The plot methods only auto-fit ``y_name`` when it is ``None``; setting it
        back to ``None`` here whenever ``selection`` differs from the last render
        makes a new event/trace/trial choice refit to that data, while a manual
        zoom or typed range persists across renders that keep the same selection.
        """
        if self._selection_keys.get(plot_key) != selection:
            self._selection_keys[plot_key] = selection
            setattr(self, y_name, None)

    @staticmethod
    def _render_to_bytes(plot: object, save_format: str) -> BytesIO:  # pragma: no cover - drives headless Chrome
        """Render a plot to an in-memory image buffer in the requested format.

        Rendering rasterizes through headless Chrome (via Selenium), the same
        mechanism Bokeh's file exporters use, but the bytes are returned instead
        of written to disk so a browser download can stream them to wherever the
        user chooses to save.

        Parameters
        ----------
        plot : holoviews.Element
            The plot to render.
        save_format : str
            Either ``"png"`` or ``"svg"``.

        Returns
        -------
        io.BytesIO
            Buffer positioned at the start, holding the encoded image.
        """
        buffer = BytesIO()
        with selenium.webdriver.Chrome(options=_headless_chrome_options()) as webdriver:
            rendered = hv.render(plot, backend="bokeh")
            if save_format == "svg":
                rendered.output_backend = "svg"
                buffer.write(get_svgs(rendered, driver=webdriver)[0].encode("utf-8"))
            else:
                get_screenshot_as_png(rendered, driver=webdriver).save(buffer, format="PNG")
        buffer.seek(0)
        return buffer

    @staticmethod
    def _notify(level: str, message: str) -> None:
        """Surface a message in the Panel UI, when notifications are available.

        Notifications are ``None`` in headless/test contexts, so this is a no-op
        there. Errors are shown persistently (``duration=0``); other levels use
        Panel's default auto-dismiss.
        """
        notifications = pn.state.notifications
        if notifications is None:
            return
        if level == "error":
            notifications.error(message, duration=0)
        else:
            getattr(notifications, level)(message)

    def _render_download(self, plot: object, save_format: str, *, nothing_selected_message: str) -> BytesIO:
        """Render ``plot`` for a browser download, reporting problems as notifications.

        Returns an empty buffer (so the download simply yields nothing) when there
        is no plot to save or the Chrome render fails; either way the reason is
        surfaced to the user instead of failing silently.
        """
        if plot is None:
            self._notify("warning", nothing_selected_message)
            return BytesIO()
        # Guard the render so _range_sync_hook ignores the off-screen export figure
        # and leaves the live figure (and its number-box controls) intact.
        self._exporting = True
        try:
            return self._render_to_bytes(plot, save_format)
        except Exception as error:
            self._notify("error", f"Could not save plot: {error}")
            return BytesIO()
        finally:
            self._exporting = False

    # The download_* methods back the "Save As…" FileDownload buttons. Each
    # re-renders its plot so the download reflects the current zoom/pan/limits
    # rather than the element cached at the last reactive render (the live range
    # params are deliberately not allowed to re-trigger those renders).
    def download_heatmap(self) -> BytesIO:
        """Return the current heatmap view as an image buffer for browser download."""
        return self._render_download(self.heatmap(), self.save_options_heatmap, nothing_selected_message="")

    def download_cont(self) -> BytesIO:
        """Return the current single-event PSTH view as an image buffer for browser download."""
        return self._render_download(self.contPlot(), self.save_options_cont, nothing_selected_message="")

    def download_overlay(self) -> BytesIO:
        """Return the current multi-event comparison view as an image buffer for browser download."""
        return self._render_download(
            self.update_selector(),
            self.save_options_overlay,
            nothing_selected_message="No events selected — nothing to save.",
        )

    def download_trials(self) -> BytesIO:
        """Return the current selected-trials view as an image buffer for browser download."""
        return self._render_download(
            self.plot_specific_trials(),
            self.save_options_trials,
            nothing_selected_message="No trials selected — nothing to save.",
        )

    # function to change Y values based on event selection
    @param.depends("event_selector", watch=True)
    def _update_x_y(self) -> None:
        x_value = self.columns_dict[self.event_selector]
        y_value = self.columns_dict[self.event_selector]
        self.param["x"].objects = [x_value[-4]]
        self.param["y"].objects = overview_y_options(y_value)
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

    def overlay_effective_colors(self) -> dict[str, str]:
        """Return the effective comparison-plot colour for each selected event.

        The colour for an event is its explicit override from
        ``overlay_color_overrides`` when present, otherwise the palette colour at
        that event's position in the current selection (cycling through
        ``overlay_palette`` so long selections wrap around).

        Returns
        -------
        dict of str to str
            Maps each entry of ``selector_for_multipe_events_plot`` to a hex
            colour string, preserving the selection order.
        """
        base_colors = hv.Cycle(self.overlay_palette).values
        return {
            event: self.overlay_color_overrides.get(event, base_colors[i % len(base_colors)])
            for i, event in enumerate(self.selector_for_multipe_events_plot)
        }

    @param.depends(
        "selector_for_multipe_events_plot",
        "Y_Label",
        "overlay_palette",
        "overlay_color_overrides",
        "Height_Plot",
        "Width_Plot",
        "hide_minor_ticks",
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
            self._reset_y_on_selection_change("overlay", "overlay_Y", tuple(arr))
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
            # Colour each event's curve (and its matching spread) from its effective
            # colour (per-event override, else the chosen palette). Curves and spreads
            # are built in the same selection order, so a positional cycle over this
            # list assigns consistent per-event colours.
            effective_colors = self.overlay_effective_colors()
            palette_colors = hv.Cycle([effective_colors[event] for event in arr])
            plot_combine = (
                (overlay * spread)
                .opts(opts.NdOverlay(xlabel="Time (s)", ylabel=self.Y_Label))
                .opts(opts.Curve(color=palette_colors), opts.Spread(fill_color=palette_colors))
                .opts(shared_axes=False)
            )
            op = make_dir(self.filepath)
            op_filename = os.path.join(op, str(arr) + "_mean")

            plot_combine = plot_combine.opts(
                hooks=[
                    self._range_sync_hook("overlay", "overlay_X", "overlay_Y"),
                    self._hide_minor_ticks_hook("hide_minor_ticks"),
                ]
            )
            self.results_psth["plot_combine"] = plot_combine
            self.results_psth["op_combine"] = op_filename
            return plot_combine

    # function to plot mean PSTH, single trial in PSTH and all the trials of PSTH with mean
    @param.depends(
        "event_selector",
        "x",
        "y",
        "Y_Label",
        "Height_Plot",
        "Width_Plot",
        "trace_color",
        "hide_minor_ticks",
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
        self._reset_y_on_selection_change("cont", "cont_Y", (self.event_selector, self.y))
        if self.y == "All":
            if self.cont_Y is None:
                self.cont_Y = (np.nanmin(np.asarray(df1)) - 0.5, np.nanmax(np.asarray(df1)) - 0.5)

            # Individual trial columns come from the data (the y-selector lists only
            # overview views), i.e. everything left after dropping bins and the mean.
            bin_regex = re.compile("bin_")
            trial_cols = [c for c in remove_cols(list(df1.columns)) if c != "mean" and not bin_regex.match(c)]

            ndoverlay = hv.NdOverlay({c: hv.Curve((df1[self.x], df1[c])) for c in trial_cols})
            img1 = datashade(ndoverlay, normalization="linear", aggregator=ds.count())
            x_points = df1[self.x]
            y_points = df1["mean"]
            img2 = hv.Curve((x_points, y_points))
            img = (img1 * img2).opts(
                opts.Curve(
                    width=int(self.Width_Plot),
                    height=int(self.Height_Plot),
                    line_width=4,
                    color=self.trace_color,
                    xlim=self.cont_X,
                    ylim=self.cont_Y,
                    xlabel="Time (s)",
                    ylabel=self.Y_Label,
                )
            )

            img = img.opts(
                hooks=[
                    self._range_sync_hook("cont", "cont_X", "cont_Y"),
                    self._hide_minor_ticks_hook("hide_minor_ticks"),
                ]
            )
            op = make_dir(self.filepath)
            op_filename = os.path.join(op, self.event_selector + "_" + self.y)
            self.results_psth["plot"] = img
            self.results_psth["op"] = op_filename

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

            plot_curve = hv.Curve((xpoints[index], ypoints[index]))
            plot_spread = hv.Spread((xpoints[index], ypoints[index], err[index], err[index]))
            plot = (plot_curve * plot_spread).opts({"Curve": ropts_curve, "Spread": ropts_spread})
            plot = plot.opts(
                hooks=[
                    self._range_sync_hook("cont", "cont_X", "cont_Y"),
                    self._hide_minor_ticks_hook("hide_minor_ticks"),
                ]
            )
            op = make_dir(self.filepath)
            op_filename = os.path.join(op, self.event_selector + "_" + self.y)
            self.results_psth["plot"] = plot
            self.results_psth["op"] = op_filename

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
            plot = plot.opts(
                hooks=[
                    self._range_sync_hook("cont", "cont_X", "cont_Y"),
                    self._hide_minor_ticks_hook("hide_minor_ticks"),
                ]
            )
            op = make_dir(self.filepath)
            op_filename = os.path.join(op, self.event_selector + "_" + self.y)
            self.results_psth["plot"] = plot
            self.results_psth["op"] = op_filename

            return plot

    # function to plot specific PSTH trials
    @param.depends(
        "event_selector",
        "x",
        "psth_y",
        "select_trials_checkbox",
        "Y_Label",
        "Height_Plot",
        "Width_Plot",
        "trace_color",
        "mean_color",
        "hide_minor_ticks",
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

        if not self.psth_y:
            return None
        selected_trials = [s.split(" - ")[1] for s in list(self.psth_y)]

        # Refit the y-axis whenever the event / trial selection / display mode changes.
        self._reset_y_on_selection_change(
            "trials", "trials_Y", (self.event_selector, tuple(self.psth_y), tuple(self.select_trials_checkbox))
        )
        if self.trials_Y is None:
            selected = np.asarray(df_psth[selected_trials])
            self.trials_Y = (np.nanmin(selected) - 0.5, np.nanmax(selected) + 0.5)

        timestamps = df_psth["timestamps"]
        index = np.arange(0, timestamps.shape[0], 3)

        show_mean = "mean" in self.select_trials_checkbox
        # Fall back to individual trials when nothing is checked so the plot is never blank.
        show_trials = ("just trials" in self.select_trials_checkbox) or not show_mean

        axis_opts = dict(
            width=int(self.Width_Plot),
            height=int(self.Height_Plot),
            xlim=self.trials_X,
            ylim=self.trials_Y,
            xlabel="Time (s)",
            ylabel=self.Y_Label,
        )

        layers = []
        if show_trials:
            overlay = hv.NdOverlay(
                {c: hv.Curve((timestamps[index], df_psth[c][index]), kdims=["Time (s)"]) for c in selected_trials}
            )
            layers.append(overlay.opts(**axis_opts))
        if show_mean:
            arr = np.asarray(df_psth[selected_trials])
            mean = np.nanmean(arr, axis=1)
            err = np.nanstd(arr, axis=1) / math.sqrt(arr.shape[1])
            # Contrast the mean against the trials when both are shown; otherwise use the trace colour.
            mean_curve_color = self.mean_color if show_trials else self.trace_color
            ropts_curve = dict(color=mean_curve_color, **axis_opts)
            ropts_spread = dict(
                width=int(self.Width_Plot),
                height=int(self.Height_Plot),
                fill_alpha=0.3,
                fill_color=mean_curve_color,
                line_width=0,
            )
            plot_curve = hv.Curve((timestamps[index], mean[index]))
            plot_spread = hv.Spread((timestamps[index], mean[index], err[index], err[index]))
            layers.append((plot_curve * plot_spread).opts({"Curve": ropts_curve, "Spread": ropts_spread}))

        result = layers[0]
        for extra_layer in layers[1:]:
            result = result * extra_layer
        result = result.opts(
            hooks=[
                self._range_sync_hook("trials", "trials_X", "trials_Y"),
                self._hide_minor_ticks_hook("hide_minor_ticks"),
            ]
        )

        op = make_dir(self.filepath)
        op_filename = os.path.join(op, self.event_selector + "_selected_trials")
        self.results_psth["trials"] = result
        self.results_psth["op_trials"] = op_filename
        return result

    # function to show heatmaps for each event
    @param.depends(
        "event_selector_heatmap",
        "color_map",
        "height_heatmap",
        "width_heatmap",
        "heatmap_y",
        "heatmap_clim",
        "hide_minor_ticks_heatmap",
        "ticks_inside_heatmap",
        "hide_outer_border_heatmap",
    )
    def heatmap(self) -> hv.Element:
        """Render a trial heatmap for the selected event.

        Returns
        -------
        holoviews.Element
            A ``QuadMesh`` (single trial) or a datashaded ``QuadMesh`` overlay
            (multiple trials), coloured by the selected colour map.
        """
        # Refit the colour scale whenever the event or trial selection changes; a
        # manual clim persists across renders that keep the same selection (same
        # convention as the PSTH Y-range auto-fit). The Trials axis has no user
        # control, so it needs no reset.
        selection = (self.event_selector_heatmap, tuple(self.heatmap_y))
        self._reset_y_on_selection_change("heatmap_clim", "heatmap_clim", selection)

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

        # Auto-fit the colour scale to the current selection on first render (or
        # after a selection change reset it above). ``trial_numbers`` is captured
        # before the single-trial duplication below so the Y range and ticks reflect
        # the real trials, and its 0.5 padding gives the lone-trial case a
        # non-degenerate range.
        trial_numbers = event_ts_for_each_event
        if self.heatmap_clim is None:
            self.heatmap_clim = (float(np.nanmin(z_score)), float(np.nanmax(z_score)))
        clim = self.heatmap_clim
        # Always span the full cell edges so the first and last trial rows render at
        # full height; recomputed every render so no stale/zoomed range can clip them.
        ylim = (float(trial_numbers.min()) - 0.5, float(trial_numbers.max()) + 0.5)
        font_size = {"labels": 16, "yticks": 6}

        # A single-trial heatmap (e.g. a group average with only one contributing
        # session) cannot be datashaded (datashade needs >=2 rows to infer cell
        # edges) and, drawn as a raw QuadMesh over the full ~30k-sample time axis,
        # overflows Bokeh's client-side renderer ("Maximum call stack size
        # exceeded"), which blanks the entire dashboard. Duplicate the lone trial
        # into a 2-row mesh so it flows through the same datashaded path as the
        # multi-trial case and renders as a single uniform band; ``yticks`` (kept
        # from before the duplication) still shows the one trial.
        if event_ts_for_each_event.shape[0] == 1:
            z_score = np.vstack([z_score, z_score])
            event_ts_for_each_event = np.array([1, 2])

        ropts = dict(
            width=int(width),
            height=int(height),
            ylabel="Trials",
            xlabel="Time (s)",
            fontsize=font_size,
            yticks=yticks,
            invert_yaxis=True,
            xlim=self.heatmap_X,
            ylim=ylim,
        )
        dummy_image = hv.QuadMesh((time[0:100], event_ts_for_each_event, z_score[:, 0:100])).opts(
            colorbar=True, cmap=process_cmap(self.color_map, provider="matplotlib"), clim=clim
        )
        actual_image = hv.QuadMesh((time, event_ts_for_each_event, z_score))

        # Shade the data with the SAME linear colour limits as the dummy colorbar so
        # the legend actually matches the pixels. Without clims/cnorm here, datashade
        # auto-normalizes the data to its own min/max (and eq-hist), so the colour-scale
        # boxes would move only the colorbar and never recolour the data.
        dynspread_img = datashade(
            actual_image,
            cmap=process_cmap(self.color_map, provider="matplotlib"),
            cnorm="linear",
            clims=clim,
        ).opts(**ropts)
        image = ((dummy_image * dynspread_img).opts(opts.QuadMesh(width=int(width), height=int(height)))).opts(
            shared_axes=False
        )
        image = image.opts(
            hooks=[
                self._range_sync_hook("heatmap", "heatmap_X"),
                self._hide_minor_ticks_hook("hide_minor_ticks_heatmap"),
                self._ticks_inside_hook("ticks_inside_heatmap"),
                self._hide_outer_border_hook("hide_outer_border_heatmap"),
            ]
        )

        op = make_dir(self.filepath)
        op_filename = os.path.join(op, self.event_selector_heatmap + "_" + "heatmap")
        self.results_hm["plot"] = image
        self.results_hm["op"] = op_filename

        return image
