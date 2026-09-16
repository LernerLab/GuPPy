"""Panel components for searching the DANDI Archive and previewing what an NWB asset holds."""

import logging
from threading import Thread

# holoviews must be imported before the first pn.extension() call so Panel wires up the HoloViews
# bokeh opts namespace the trace preview's opts rely on. Mirrors the import-then-extension
# ordering in custom_events_config.py.
import holoviews as hv
import pandas as pd
import panel as pn

from ..utils.dandi_catalog import (
    PHOTOMETRY_SEARCH_TERMS,
    AssetPreview,
    DandisetReference,
    DandisetSummary,
    PhotometryProbe,
    PhotometryVerdictCache,
    collect_filter_options,
    filter_dandisets,
    format_byte_size,
    list_dandiset_references,
    list_nwb_assets,
    order_for_crawl,
    preview_asset,
    search_dandisets,
    summarize_dandisets,
    verify_dandisets,
)

logger = logging.getLogger(__name__)

pn.extension()
hv.extension("bokeh")

# Width of the browser's widgets inside the 1000px Input Folder Selection card.
BROWSER_WIDTH = 950

CATALOG_COLUMNS = (
    "Dandiset",
    "Name",
    "Species",
    "Subjects",
    "Files",
    "Size",
    "Brain regions",
    "Indicators",
)
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


def catalog_dataframe(summaries: list[DandisetSummary]) -> pd.DataFrame:
    """Lay a list of dandiset summaries out as the catalog table's rows.

    Parameters
    ----------
    summaries : list of DandisetSummary
        Rows to render, in table order.

    Returns
    -------
    pandas.DataFrame
        One row per summary, with the columns in :data:`CATALOG_COLUMNS`.
    """
    rows = [
        {
            "Dandiset": summary.identifier,
            "Name": summary.name,
            "Species": ", ".join(summary.species),
            "Subjects": summary.subject_count,
            "Files": summary.file_count,
            "Size": format_byte_size(summary.size_in_bytes),
            "Brain regions": ", ".join(summary.brain_regions),
            "Indicators": ", ".join(summary.indicators),
        }
        for summary in summaries
    ]
    return pd.DataFrame(rows, columns=list(CATALOG_COLUMNS))


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


def describe_dandiset(summary: DandisetSummary) -> str:
    """Render a dandiset summary as the Markdown shown when its catalog row is selected.

    Parameters
    ----------
    summary : DandisetSummary
        The dandiset to describe.

    Returns
    -------
    str
        Markdown giving the dandiset's identity, scale, detected regions and indicators,
        keywords, credits and abstract.
    """
    contributors = ", ".join(summary.contributors[:4])
    if len(summary.contributors) > 4:
        contributors += ", et al."
    # DANDI records licenses as SPDX URIs ("spdx:CC-BY-4.0"); the identifier alone reads better.
    licenses = ", ".join(term.removeprefix("spdx:") for term in summary.license_terms)
    lines = [
        f"#### {summary.identifier} — {summary.name}",
        f"[Open {summary.identifier} on dandiarchive.org]({summary.url}) · version `{summary.version}`"
        f" · {licenses or 'license not stated'}",
        f"**Species:** {', '.join(summary.species) or '—'} · **Subjects:** {summary.subject_count or '—'}"
        f" · **Files:** {summary.file_count} · **Size:** {format_byte_size(summary.size_in_bytes)}",
        f"**Brain regions:** {', '.join(summary.brain_regions) or '—'}",
        f"**Indicators:** {', '.join(summary.indicators) or '—'}",
        f"**Approaches:** {', '.join(summary.approaches) or '—'}",
        f"**Keywords:** {', '.join(summary.keywords) or '—'}",
        f"**Contributors:** {contributors or '—'}",
        "",
        summary.description,
    ]
    return "\n\n".join(lines)


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
        width=BROWSER_WIDTH - 120,
        height=280,
        legend_position="right",
        title=f"First {traces.timestamps[-1]:.0f} s of {traces.series_name}",
    )


class PhotometryPreviewPane:
    """The shared preview of one NWB asset: what it holds, and the start of its traces.

    Renders an :class:`~guppy.utils.dandi_catalog.AssetPreview` as a Markdown summary, a
    per-channel table, and an overlay of the opening seconds of every channel of one response
    series. A file that stores each channel as its own series gets a series picker, which
    re-streams the chosen one. Used both by the catalog browser (on a representative file of
    the selected dandiset) and by the asset selector (on the file the user picked).

    Parameters
    ----------
    preview_function : callable, optional
        Injection point for the streaming preview, used when the series picker changes;
        defaults to :func:`~guppy.utils.dandi_catalog.preview_asset`.
    width : int, optional
        Fixed width of the composed layout, in pixels.

    Attributes
    ----------
    panel : panel.Column
        The composed layout, hidden until a preview is shown.
    preview : AssetPreview or None
        The preview currently on display, or None before the first one.
    """

    def __init__(self, *, preview_function: object = preview_asset, width: int = BROWSER_WIDTH) -> None:
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


class DandiBrowser:
    """A searchable catalog of the DANDI Archive's fiber photometry dandisets.

    Runs the archive's full-text search over the photometry terms (or over the user's own
    query), summarizes every hit, and lists them in a sortable table. The catalog is held in
    the component, so the region, indicator, species, approach and scale filters narrow it
    without going back to the network. Selecting a row shows that dandiset's metadata, and
    an explicit inspect action streams one of its NWB files to report the channels it holds
    and plot the start of each trace.

    Parameters
    ----------
    on_dandiset_selected : callable or None, optional
        Called with the six-digit identifier when the user chooses a dandiset to analyze.
    search_function : callable, optional
        Injection point for the catalog search; defaults to
        :func:`~guppy.utils.dandi_catalog.search_dandisets`.
    list_assets_function : callable, optional
        Injection point for the asset listing; defaults to
        :func:`~guppy.utils.dandi_catalog.list_nwb_assets`.
    preview_function : callable, optional
        Injection point for the streaming preview; defaults to
        :func:`~guppy.utils.dandi_catalog.preview_asset`.
    width : int, optional
        Fixed width of the composed layout, in pixels.

    Attributes
    ----------
    panel : panel.Column
        The composed Panel layout to embed in a page.
    summaries : list of DandisetSummary
        The catalog as last fetched, before filtering.
    visible_summaries : list of DandisetSummary
        The rows the filters currently leave in the table.
    selected_summary : DandisetSummary or None
        The dandiset whose row is selected, or None when no row is.
    """

    def __init__(
        self,
        *,
        on_dandiset_selected: object = None,
        search_function: object = search_dandisets,
        list_assets_function: object = list_nwb_assets,
        preview_function: object = preview_asset,
        verify_function: object = verify_dandisets,
        references_function: object = list_dandiset_references,
        summarize_function: object = summarize_dandisets,
        verdict_cache: PhotometryVerdictCache | None = None,
        width: int = BROWSER_WIDTH,
    ) -> None:
        self.on_dandiset_selected = on_dandiset_selected
        self.search_function = search_function
        self.list_assets_function = list_assets_function
        self.preview_function = preview_function
        self.verify_function = verify_function
        self.references_function = references_function
        self.summarize_function = summarize_function
        self.verdict_cache = verdict_cache if verdict_cache is not None else PhotometryVerdictCache()
        self.width = width

        # Summaries of the dandisets verification has confirmed hold photometry, which is
        # what the table draws from: a row only appears once something has read the files.
        self.summaries: list[DandisetSummary] = []
        self.visible_summaries: list[DandisetSummary] = []
        # State of the verification or crawl currently running, if any.
        self._verification: dict[str, object] = {}
        # Set while the filter options are rewritten after a search, so dropping a value that
        # the new catalog no longer offers does not re-run the filters mid-rewrite.
        self._rewriting_filter_options = False

        self.query_input = pn.widgets.TextInput(
            name="Search terms",
            placeholder="e.g. dopamine striatum",
            width=360,
        )
        self.photometry_only = pn.widgets.Checkbox(
            name="Fiber photometry datasets only",
            value=True,
        )
        self.search_button = pn.widgets.Button(name="Search DANDI", button_type="primary", width=150)
        self.search_button.on_click(self.refresh_catalog)

        self.crawl_button = pn.widgets.Button(name="Search every dandiset", width=220)
        self.crawl_button.on_click(self.crawl_archive)
        self.stop_button = pn.widgets.Button(name="Stop", button_type="warning", width=90, visible=False)
        self.stop_button.on_click(self.stop_verification)
        self.verification_progress = pn.indicators.Progress(name="Verifying", value=0, max=1, width=460, visible=False)

        self.species_filter = pn.widgets.MultiChoice(name="Species", options=[], width=300)
        self.brain_region_filter = pn.widgets.MultiChoice(name="Brain region", options=[], width=300)
        self.indicator_filter = pn.widgets.MultiChoice(name="Indicator", options=[], width=300)
        self.approach_filter = pn.widgets.MultiChoice(name="Approach / technique", options=[], width=300)
        self.minimum_subjects = pn.widgets.IntInput(name="Min. subjects", value=0, start=0, width=140)
        self.minimum_files = pn.widgets.IntInput(name="Min. NWB files", value=0, start=0, width=140)
        self.published_only = pn.widgets.Checkbox(name="Published versions only", value=False)
        for widget in (
            self.query_input,
            self.photometry_only,
            self.species_filter,
            self.brain_region_filter,
            self.indicator_filter,
            self.approach_filter,
            self.minimum_subjects,
            self.minimum_files,
            self.published_only,
        ):
            widget.param.watch(self.apply_filters, "value")

        self.status = pn.pane.Markdown(
            "Press **Search DANDI** to load the archive's fiber photometry dandisets.",
            width=width,
        )
        self.results_table = pn.widgets.Tabulator(
            catalog_dataframe([]),
            show_index=False,
            disabled=True,
            selectable=1,
            width=width,
            height=320,
            widths={"Name": 320},
            # Names and region lists outrun their columns; a hover tooltip shows the whole
            # cell without widening the table past the card.
            configuration={"columnDefaults": {"tooltip": True}},
        )
        self.results_table.param.watch(self._on_row_selected, "selection")

        self.dandiset_details = pn.pane.Markdown("", width=width)
        self.use_button = pn.widgets.Button(name="Analyze this dandiset", button_type="success", width=200)
        self.use_button.on_click(self._on_use_clicked)
        self.inspect_button = pn.widgets.Button(name="Inspect largest NWB file", width=220)
        self.inspect_button.on_click(self.inspect_selected_dandiset)
        self.action_row = pn.Row(self.use_button, self.inspect_button, visible=False)

        self.preview_pane = PhotometryPreviewPane(preview_function=preview_function, width=width)

        self.panel = pn.Column(
            pn.Row(self.query_input, pn.Column(pn.Spacer(height=22), self.search_button)),
            pn.Row(self.photometry_only, self.published_only),
            pn.Row(self.species_filter, self.brain_region_filter),
            pn.Row(self.indicator_filter, self.approach_filter),
            pn.Row(self.minimum_subjects, self.minimum_files),
            pn.Row(self.crawl_button, self.stop_button, self.verification_progress),
            self.status,
            self.results_table,
            self.dandiset_details,
            self.action_row,
            self.preview_pane.panel,
            width=width,
        )

    # ------------------------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------------------------

    def refresh_catalog(self, event: object = None) -> None:
        """Search the archive and verify what it returned.

        With **Fiber photometry datasets only** checked the archive is searched for the
        photometry terms and each hit is then read, so that every row in the table is a
        dandiset something has confirmed holds fiber photometry GuPPy can read. Unchecked,
        the query goes to the archive itself and its results are listed as they come, since
        the user is then browsing DANDI rather than the photometry catalog.

        Parameters
        ----------
        event : object, optional
            The Panel click event; unused.
        """
        query = self.query_input.value.strip()
        if not self.photometry_only.value:
            if not query:
                self.status.object = (
                    "⚠️ Searching the whole archive needs a search term. Type one, or tick "
                    "**Fiber photometry datasets only** to load the photometry catalog."
                )
                return
            self.status.object = "Searching the DANDI Archive…"
            self.summaries = self.search_function(terms=(query,))
            self._rewrite_filter_options()
            self.apply_filters()
            return

        self.status.object = "Searching the DANDI Archive…"
        candidates = [summary for summary in self.search_function(terms=PHOTOMETRY_SEARCH_TERMS) if summary.file_count]
        self.summaries = []
        self.apply_filters()
        self._start_verification(
            references=[DandisetReference.from_summary(summary) for summary in candidates],
            known_summaries={summary.identifier: summary for summary in candidates},
            description="candidate",
        )

    def crawl_archive(self, event: object = None) -> None:
        """Read every dandiset on the archive, not only those whose text mentions photometry.

        The search terms find datasets that say what they are; this finds the rest. It reads
        every asset of every dandiset, which takes far longer, and reports each dandiset as
        soon as it settles rather than at the end.

        Parameters
        ----------
        event : object, optional
            The Panel click event; unused.
        """
        self.status.object = "Listing every dandiset on the archive…"
        references = self.references_function()
        found_by_text = [summary.identifier for summary in self.summaries]
        self._start_verification(
            references=order_for_crawl(references, first=found_by_text),
            known_summaries={summary.identifier: summary for summary in self.summaries},
            description="dandiset",
        )

    def stop_verification(self, event: object = None) -> None:
        """Ask the running verification to stop after the dandiset it is on.

        Parameters
        ----------
        event : object, optional
            The Panel click event; unused.
        """
        self._verification["stopping"] = True
        self.stop_button.disabled = True

    def _start_verification(self, *, references: list, known_summaries: dict, description: str) -> None:
        """Verify ``references`` on a worker thread, adding each dandiset as it is confirmed.

        The work runs off the server IOLoop and is polled back onto it, because a run can
        last from seconds to an hour and blocking the loop for that would drop the browser's
        websocket. Confirmations arrive early and rejections late -- ruling a dandiset out
        means reading every asset it has -- so the table fills quickly and then slows.
        """
        total = len(references)
        if not total:
            self.status.object = "No dandisets to verify."
            return
        self.search_button.disabled = True
        self.crawl_button.disabled = True
        self.stop_button.disabled = False
        self.stop_button.visible = True
        self.verification_progress.max = total
        self.verification_progress.value = 0
        self.verification_progress.visible = True

        self._verification = {
            "settled": 0,
            "total": total,
            "confirmed": [],
            "known": known_summaries,
            "description": description,
            "stopping": False,
        }

        def on_verdict(reference: object, holds: bool) -> None:
            state = self._verification
            state["settled"] += 1
            if holds:
                state["confirmed"].append(reference.identifier)

        def worker() -> None:
            self.verify_function(
                references,
                cache=self.verdict_cache,
                on_verdict=on_verdict,
                should_stop=lambda: bool(self._verification["stopping"]),
            )

        self._verification["thread"] = Thread(target=worker)
        self._verification["thread"].start()
        self._verification["callback"] = pn.state.add_periodic_callback(self._poll_verification, period=500)

    def _poll_verification(self) -> None:
        """Draw the running verification's progress, folding in whatever it has confirmed."""
        state = self._verification
        settled, total = state["settled"], state["total"]
        self.verification_progress.value = min(settled, total)
        confirmed = list(state["confirmed"])
        # Completion is the worker thread finishing rather than the count reaching the total,
        # which it does not when the run is stopped early.
        finished = not state["thread"].is_alive()
        if finished or len(confirmed) != len(self.summaries):
            # Redraws the table, and rewrites the status line, so the progress report below
            # has to come after it.
            self._show_confirmed(confirmed)
        if finished:
            state["callback"].stop()
            self.search_button.disabled = False
            self.crawl_button.disabled = False
            self.stop_button.visible = False
            self.verification_progress.visible = False
        self.status.object = self._verification_status(
            settled=settled, total=total, confirmed=len(confirmed), finished=finished
        )

    def _verification_status(self, *, settled: int, total: int, confirmed: int, finished: bool) -> str:
        """Describe how far verification has got and what its answer does not cover."""
        noun = self._verification["description"]
        holds = "holds" if confirmed == 1 else "hold"
        if not finished:
            return f"Read **{settled}** of {total} {noun}(s) so far — " f"**{confirmed}** {holds} fiber photometry."
        stopped = bool(self._verification["stopping"])
        lead = "Stopped after" if stopped else "Read"
        caveat = (
            ""
            if noun == "dandiset" and not stopped
            else (
                " Datasets whose description never mentions photometry are not in this list; "
                "**Search every dandiset** reads the rest of the archive to find them."
            )
        )
        return f"{lead} **{settled}** of {total} {noun}(s): **{confirmed}** {holds} fiber photometry.{caveat}"

    def _show_confirmed(self, identifiers: list) -> None:
        """Put the confirmed dandisets into the catalog, fetching any summary not yet held."""
        known = self._verification["known"]
        missing = [identifier for identifier in identifiers if identifier not in known]
        if missing:
            for summary in self.summarize_function(missing):
                known[summary.identifier] = summary
        self.summaries = [known[identifier] for identifier in identifiers if identifier in known]
        self._rewrite_filter_options()
        self.apply_filters()

    def _rewrite_filter_options(self) -> None:
        """Repoint each categorical filter at the values the new catalog actually holds."""
        options = collect_filter_options(self.summaries)
        self._rewriting_filter_options = True
        try:
            for widget, key in (
                (self.species_filter, "species"),
                (self.brain_region_filter, "brain_regions"),
                (self.indicator_filter, "indicators"),
                (self.approach_filter, "approaches"),
            ):
                widget.options = options[key]
                widget.value = [value for value in widget.value if value in options[key]]
        finally:
            self._rewriting_filter_options = False

    def apply_filters(self, event: object = None) -> None:
        """Narrow the catalog by the current filter values and redraw the table.

        Parameters
        ----------
        event : object, optional
            The Panel value-change event; unused.
        """
        if self._rewriting_filter_options:
            return
        # The query box searches the archive itself when the photometry catalog is switched
        # off, so filtering on it locally as well would hide rows the archive matched on
        # fields the summary does not carry.
        local_query = self.query_input.value if self.photometry_only.value else ""
        self.visible_summaries = filter_dandisets(
            self.summaries,
            query=local_query,
            species=self.species_filter.value,
            brain_regions=self.brain_region_filter.value,
            indicators=self.indicator_filter.value,
            approaches=self.approach_filter.value,
            minimum_subjects=self.minimum_subjects.value,
            minimum_files=self.minimum_files.value,
            published_only=self.published_only.value,
        )
        self.results_table.value = catalog_dataframe(self.visible_summaries)
        self.results_table.selection = []
        self._clear_selection()
        self.status.object = (
            f"Showing **{len(self.visible_summaries)}** of {len(self.summaries)} dandiset(s). "
            "Select a row to see its metadata."
        )

    # ------------------------------------------------------------------------------------
    # Selection and preview
    # ------------------------------------------------------------------------------------

    @property
    def selected_summary(self) -> DandisetSummary | None:
        """Return the summary of the selected catalog row, or None when nothing is selected."""
        selection = self.results_table.selection
        if not selection:
            return None
        return self.visible_summaries[selection[0]]

    def _clear_selection(self) -> None:
        self.dandiset_details.object = ""
        self.action_row.visible = False
        self.preview_pane.clear()

    def _on_row_selected(self, event: object) -> None:
        summary = self.selected_summary
        if summary is None:
            self._clear_selection()
            return
        self.dandiset_details.object = describe_dandiset(summary)
        self.action_row.visible = True
        self.preview_pane.clear()

    def _on_use_clicked(self, event: object = None) -> None:
        summary = self.selected_summary
        if summary is None:
            self.status.object = "⚠️ Select a dandiset row first."
            return
        logger.info("DANDI browser: dandiset %s chosen for analysis", summary.identifier)
        if self.on_dandiset_selected is not None:
            self.on_dandiset_selected(summary.identifier)

    def inspect_selected_dandiset(self, event: object = None) -> None:
        """Stream the selected dandiset's largest NWB file and preview what it holds.

        The largest file is the representative one: within a dandiset the recordings dwarf
        the behavior-only files, so the biggest asset is the one that carries traces.

        Parameters
        ----------
        event : object, optional
            The Panel click event; unused.
        """
        summary = self.selected_summary
        if summary is None:
            self.status.object = "⚠️ Select a dandiset row first."
            return
        # Assets are listed from the draft version, which is the version GuPPy streams from.
        assets = self.list_assets_function(dandiset_id=summary.identifier)
        if not assets:
            self.status.object = f"⚠️ Dandiset {summary.identifier} holds no NWB assets."
            self.preview_pane.clear()
            return
        largest = max(assets, key=lambda asset: asset.size_in_bytes)
        self.status.object = (
            f"Streaming `{largest.path}` ({format_byte_size(largest.size_in_bytes)}) "
            f"from dandiset {summary.identifier}…"
        )
        preview = self.preview_function(dandiset_id=summary.identifier, asset_path=largest.path)
        self.preview_pane.show(preview=preview)
        self.status.object = (
            f"Inspected `{largest.path}`, the largest of {len(assets)} NWB asset(s) in dandiset "
            f"{summary.identifier}."
        )
