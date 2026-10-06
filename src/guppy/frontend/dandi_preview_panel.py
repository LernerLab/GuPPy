"""Panel component that renders what one DANDI NWB asset holds.

Given an :class:`~guppy.utils.dandi_preview.AssetPreview`, it shows the file's channels, the
recording sites and indicators behind them, the store names Step 1 will ask for labels against,
and the first seconds of every trace. A file holding several response series gets a picker that
re-streams the chosen one.
"""

# holoviews must be imported before the first pn.extension() call so Panel wires up the HoloViews
# bokeh opts namespace the trace preview's opts rely on. Mirrors the import-then-extension
# ordering in custom_events_config.py.
import holoviews as hv
import pandas as pd
import panel as pn

from .frontend_utils import DANDI_PANEL_WIDTH
from ..utils.dandi_preview import AssetPreview, PhotometryProbe, preview_asset

pn.extension()
hv.extension("bokeh")

CHANNEL_COLUMNS = (
    "Store name",
    "Brain region",
    "Indicator",
    "Excitation (nm)",
    "Emission (nm)",
    "Suggested label",
)
# Store names and locations routinely outrun any column that fits six of them side by side, so
# the widths are set to keep every column visible and leave the overflow to the cell tooltips.
CHANNEL_COLUMN_WIDTHS = {
    "Store name": 290,
    "Brain region": 140,
    "Indicator": 105,
    "Excitation (nm)": 115,
    "Emission (nm)": 110,
    "Suggested label": 170,
}


def channel_dataframe(probe: PhotometryProbe) -> pd.DataFrame:
    """Lay a probe's channels out as the preview's channel table.

    Parameters
    ----------
    probe : PhotometryProbe
        The probed file whose channels are rendered.

    Returns
    -------
    pandas.DataFrame
        One row per channel, with the columns in :data:`CHANNEL_COLUMNS`.
    """
    rows = [
        {
            "Store name": channel.store_name,
            "Brain region": channel.location or "",
            "Indicator": channel.indicator or "",
            "Excitation (nm)": channel.excitation_wavelength_in_nm or "",
            "Emission (nm)": channel.emission_wavelength_in_nm or "",
            "Suggested label": channel.suggested_label or "",
        }
        for channel in probe.channels
    ]
    return pd.DataFrame(rows, columns=list(CHANNEL_COLUMNS))


def describe_probe(*, preview: AssetPreview) -> str:
    """Render a probed asset as the Markdown heading its channel table.

    Parameters
    ----------
    preview : AssetPreview
        The probed asset.

    Returns
    -------
    str
        Markdown naming the asset, its session and subject, and the timing of each response
        series -- or reporting that the file holds no fiber photometry data.
    """
    probe = preview.probe
    lines = [f"**File:** `{preview.asset_path}`"]
    if not probe.has_photometry:
        lines.append(
            "⚠️ This file holds no `FiberPhotometryResponseSeries`, so GuPPy has no trace to read "
            "from it. In many dandisets the small files carry only a session's behavioral events, "
            "and the recordings are the large ones."
        )
        if probe.event_names:
            lines.append(f"**Event objects:** {', '.join(probe.event_names)}")
        return "\n\n".join(lines)

    for series in probe.series:
        rate = f"{series.sampling_rate_in_hz:.2f} Hz" if series.sampling_rate_in_hz else "irregular timestamps"
        duration = f"{series.duration_in_seconds / 60:.1f} min" if series.duration_in_seconds else "unknown length"
        lines.append(
            f"**Series `{series.name}`:** {series.channel_count} channel(s) · "
            f"{series.sample_count:,} samples · {rate} · {duration}"
        )
    lines.append(f"**Event objects:** {', '.join(probe.event_names) or '—'}")
    subject = probe.subject
    subject_fields = [subject.get(key) for key in ("subject_id", "species", "sex", "age", "strain", "genotype")]
    lines.append(f"**Subject:** {' · '.join(field for field in subject_fields if field) or '—'}")
    if probe.session_description:
        lines.append(f"**Session:** {probe.session_description}")
    return "\n\n".join(lines)


def build_trace_overlay(*, preview: AssetPreview) -> hv.NdOverlay:
    """Overlay one curve per channel of a preview's example traces.

    Parameters
    ----------
    preview : AssetPreview
        A probed asset whose ``traces`` were read.

    Returns
    -------
    holoviews.NdOverlay
        One labelled curve per channel, over a shared time axis in seconds.
    """
    traces = preview.traces
    curves = {
        store_name: hv.Curve((traces.timestamps, values), "Time (s)", "Fluorescence").opts(tools=["hover"])
        for store_name, values in traces.traces.items()
    }
    return hv.NdOverlay(curves, kdims="Channel").opts(
        width=DANDI_PANEL_WIDTH - 120,
        height=280,
        legend_position="right",
        title=f"First {traces.timestamps[-1]:.0f} s of {traces.series_name}",
    )


class DandiPreviewPanel:
    """The shared preview of one NWB asset: what it holds, and the start of its traces.

    Renders an :class:`~guppy.utils.dandi_preview.AssetPreview` as a Markdown summary, a
    per-channel table, and an overlay of the opening seconds of every channel of one response
    series. A file that stores each channel as its own series gets a series picker, which
    re-streams the chosen one. Used both by the catalog browser (on a representative file of
    the selected dandiset) and by the asset selector (on the file the user picked).

    Parameters
    ----------
    preview_function : callable, optional
        Injection point for the streaming preview, used when the series picker changes;
        defaults to :func:`~guppy.utils.dandi_preview.preview_asset`.
    width : int, optional
        Fixed width of the composed layout, in pixels.

    Attributes
    ----------
    panel : panel.Column
        The composed layout, hidden until a preview is shown.
    preview : AssetPreview or None
        The preview currently on display, or None before the first one.
    """

    def __init__(self, *, preview_function: object = preview_asset, width: int = DANDI_PANEL_WIDTH) -> None:
        self.preview_function = preview_function
        self.width = width
        self.preview: AssetPreview | None = None
        # Set while the series picker's options are rewritten for a newly shown preview, so
        # repointing it does not re-stream the file the preview just read.
        self._rewriting_series_options = False

        self.summary = pn.pane.Markdown("", width=width)
        self.series_select = pn.widgets.Select(name="Traces from series", options=[], width=400)
        self.series_select.param.watch(self._on_series_change, "value")
        self.channel_table = pn.widgets.Tabulator(
            channel_dataframe(PhotometryProbe()),
            show_index=False,
            disabled=True,
            width=width,
            widths=CHANNEL_COLUMN_WIDTHS,
            configuration={"columnDefaults": {"tooltip": True}},
        )
        self.trace_pane = pn.pane.HoloViews(None, width=width)
        self.panel = pn.Column(
            self.summary,
            self.channel_table,
            self.series_select,
            self.trace_pane,
            visible=False,
        )

    def show(self, *, preview: AssetPreview) -> None:
        """Render ``preview`` into the pane and make it visible.

        Parameters
        ----------
        preview : AssetPreview
            The probed asset to display.
        """
        self.preview = preview
        self.summary.object = describe_probe(preview=preview)
        self.channel_table.value = channel_dataframe(preview.probe)
        self.channel_table.visible = preview.probe.has_photometry
        series_names = [series.name for series in preview.probe.series]
        self._rewriting_series_options = True
        try:
            self.series_select.options = series_names
            if preview.traces is not None:
                self.series_select.value = preview.traces.series_name
        finally:
            self._rewriting_series_options = False
        # A file with one series has nothing to pick between; the summary already names it.
        self.series_select.visible = len(series_names) > 1
        self.trace_pane.object = build_trace_overlay(preview=preview) if preview.traces else None
        self.panel.visible = True

    def _on_series_change(self, event: object) -> None:
        if self._rewriting_series_options or self.preview is None:
            return
        preview = self.preview_function(
            dandiset_id=self.preview.dandiset_id,
            asset_path=self.preview.asset_path,
            series_name=self.series_select.value,
        )
        self.show(preview=preview)

    def clear(self) -> None:
        """Empty the pane and hide it."""
        self.preview = None
        self.summary.object = ""
        self.channel_table.value = channel_dataframe(PhotometryProbe())
        self.trace_pane.object = None
        self.series_select.visible = False
        self.panel.visible = False
