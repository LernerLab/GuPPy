"""Panel component for finding a dandiset on the DANDI Archive.

Two screens: a searchable, filterable table of the archive's photometry datasets, and one
dandiset's own page with the action that hands its identifier on for analysis.
"""

import logging
from threading import Thread

import pandas as pd
import panel as pn

from .frontend_utils import DANDI_PANEL_WIDTH, SECONDARY_BUTTON_STYLESHEET
from ..utils.dandi_filter import (
    DandisetReference,
    PhotometryVerdictCache,
    order_for_verification,
    verify_dandisets,
)
from ..utils.dandi_search import (
    PHOTOMETRY_SEARCH_TERMS,
    DandisetSummary,
    format_byte_size,
    search_dandisets,
)

logger = logging.getLogger(__name__)

pn.extension()

# What the search box holds when the panel opens: GuPPy is for photometry, so the search it is
# for should already be running rather than waiting to be typed.
DEFAULT_SEARCH_TERM = PHOTOMETRY_SEARCH_TERMS[0]

CATALOG_COLUMNS = (
    "Dandiset",
    "Name",
    "Species",
    "Subjects",
    "Files",
    "Size",
)
# Widths summing under the table's own, so the last column ends inside the card rather than
# under its edge. Name takes whatever the others do not need, since a dandiset's title is the
# one cell that is routinely a sentence long.
CATALOG_COLUMN_WIDTHS = {
    "Dandiset": 110,
    "Name": 375,
    "Species": 145,
    "Subjects": 105,
    "Files": 90,
    "Size": 100,
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
        }
        for summary in summaries
    ]
    return pd.DataFrame(rows, columns=list(CATALOG_COLUMNS))


def describe_dandiset(summary: DandisetSummary) -> str:
    """Render a dandiset summary as the Markdown shown when its catalog row is selected.

    Parameters
    ----------
    summary : DandisetSummary
        The dandiset to describe.

    Returns
    -------
    str
        Markdown giving the dandiset's identity, scale, approaches, keywords, credits and
        abstract.
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
        f"**Approaches:** {', '.join(summary.approaches) or '—'}",
        f"**Keywords:** {', '.join(summary.keywords) or '—'}",
        f"**Contributors:** {contributors or '—'}",
        "",
        summary.description,
    ]
    return "\n\n".join(lines)


class DandiSearchPanel:
    """A searchable catalog of the DANDI Archive, on two screens.

    The list screen runs the archive's full-text search -- starting on the photometry term,
    so the panel opens on the datasets GuPPy is for -- and tabulates the hits. Selecting one
    replaces the list with that dandiset's page: its metadata, and the action that hands its
    identifier on for analysis.

    The search matches what a dataset's authors wrote about it, which is a cheap precondition
    rather than an answer. An optional filter reads the listed dandisets' NWB files and keeps
    only those holding a type GuPPy can read. That takes minutes where the search takes
    seconds, so it runs on a worker thread, is polled back onto the server IOLoop, and is off
    until asked for.

    Parameters
    ----------
    on_dandiset_selected : callable or None, optional
        Called with the six-digit identifier when the user chooses a dandiset to analyze.
    search_function : callable, optional
        Injection point for the catalog search; defaults to
        :func:`~guppy.utils.dandi_search.search_dandisets`.
    verify_function : callable, optional
        Injection point for reading the listed dandisets' files; defaults to
        :func:`~guppy.utils.dandi_filter.verify_dandisets`.
    verdict_cache : PhotometryVerdictCache or None, optional
        Where verdicts are remembered between sessions. One in GuPPy's user cache directory
        is used when not supplied.
    width : int, optional
        Fixed width of the composed layout, in pixels.

    Attributes
    ----------
    panel : panel.Column
        The composed Panel layout to embed in a page.
    summaries : list of DandisetSummary
        Every dandiset the last search returned, whether or not its files have been read.
    verdicts : dict of {str: bool or None}
        What reading the files concluded about each dandiset, empty until it is asked for.
        None means the dandiset could not be read in full.
    visible_summaries : list of DandisetSummary
        The rows currently in the table.
    selected_summary : DandisetSummary or None
        The dandiset whose row is selected, or None when no row is.
    """

    def __init__(
        self,
        *,
        on_dandiset_selected: object = None,
        search_function: object = search_dandisets,
        verify_function: object = verify_dandisets,
        verdict_cache: PhotometryVerdictCache | None = None,
        width: int = DANDI_PANEL_WIDTH,
    ) -> None:
        self.on_dandiset_selected = on_dandiset_selected
        self.search_function = search_function
        self.verify_function = verify_function
        self.verdict_cache = verdict_cache if verdict_cache is not None else PhotometryVerdictCache()
        self.width = width

        # Every dandiset the search returned, whether or not its files have been read.
        self.summaries: list[DandisetSummary] = []
        # Identifier -> whether that dandiset holds photometry, or None when it could not be
        # read. Empty until a scan has run.
        self.verdicts: dict[str, bool | None] = {}
        self.visible_summaries: list[DandisetSummary] = []
        # State of the verification or crawl currently running, if any.
        self._verification: dict[str, object] = {}
        # Whether the default search has been run; see open_catalog.
        self._searched = False

        # The archive's own text search, with the term GuPPy is for already in it. The results
        # are what a dandiset's authors wrote about it, not what its files hold -- reading the
        # files is what the thorough options below and the per-file scan are for.
        self.query_input = pn.widgets.TextInput(
            name="Search the DANDI Archive",
            value=DEFAULT_SEARCH_TERM,
            placeholder="a word, or a six-digit Dandiset ID",
            width=460,
        )
        self.query_input.param.watch(self.refresh_catalog, "enter_pressed")
        self.search_button = pn.widgets.Button(name="Search", button_type="primary", width=110)
        self.search_button.on_click(self.refresh_catalog)

        # Reads the listed dandisets' NWB files, which takes minutes where the search takes
        # seconds, so it is off until asked for.
        self.verify_listed = pn.widgets.Checkbox(
            name="Filter dandisets for GuPPy-readable fiber photometry (slow)",
            value=False,
            width=420,
        )
        self.verify_listed.param.watch(self._on_verify_listed, "value")
        self.stop_button = pn.widgets.Button(name="Stop", button_type="warning", width=90, visible=False)
        self.stop_button.on_click(self.stop_verification)
        self.verification_progress = pn.indicators.Progress(name="Reading", value=0, max=1, width=460, visible=False)

        self.status = pn.pane.Markdown("", width=width)
        self.results_table = pn.widgets.Tabulator(
            catalog_dataframe([]),
            show_index=False,
            disabled=True,
            selectable=1,
            width=width,
            height=430,
            widths=dict(CATALOG_COLUMN_WIDTHS),
            # Stretch the columns to the table's own width, so the last one ends at the edge
            # rather than leaving a dead strip beside it.
            layout="fit_columns",
            # Titles routinely outrun any width that leaves room for the other columns, so
            # they wrap onto a second line instead of being clipped mid-word.
            formatters={"Name": {"type": "textarea"}},
            configuration={"columnDefaults": {"tooltip": True}},
        )
        self.results_table.param.watch(self._on_row_selected, "selection")

        self.dandiset_details = pn.pane.Markdown("", width=width)
        self.use_button = pn.widgets.Button(name="Analyze this dandiset", button_type="success", width=200)
        self.use_button.on_click(self._on_use_clicked)
        self.back_button = pn.widgets.Button(
            name="← Back to results", width=180, stylesheets=[SECONDARY_BUTTON_STYLESHEET]
        )
        self.back_button.on_click(self.show_results)

        # Two screens rather than one growing page: a list of dandisets, or one dandiset.
        # Going back is free because the search's results are held in the component.
        self.list_view = pn.Column(
            pn.Row(
                self.query_input,
                pn.Column(pn.Spacer(height=22), self.search_button),
            ),
            pn.Row(self.verify_listed, self.stop_button),
            self.verification_progress,
            self.status,
            self.results_table,
            width=width,
        )
        self.dandiset_view = pn.Column(
            self.back_button,
            self.dandiset_details,
            self.use_button,
            width=width,
            visible=False,
        )
        self.panel = pn.Column(self.list_view, self.dandiset_view, width=width)

    # ------------------------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------------------------

    def refresh_catalog(self, event: object = None) -> None:
        """Run the archive's text search and list what it returned, reading no files.

        The box starts on the photometry term, so opening the panel already shows the
        datasets GuPPy is for. Replacing the term searches the archive for anything else,
        and a six-digit Dandiset ID finds that dandiset, which is why there is no separate
        field for one.

        Parameters
        ----------
        event : object, optional
            The Panel click or Enter event; unused.
        """
        query = self.query_input.value.strip()
        if not query:
            self.status.object = "⚠️ Type something to search for, or a six-digit Dandiset ID."
            return
        self.status.object = "Searching the DANDI Archive…"
        self._forget_verdicts()
        self.summaries = [summary for summary in self.search_function(terms=(query,)) if summary.file_count]
        self.show_results()
        self.apply_filters()

    def open_catalog(self) -> None:
        """Run the default search the first time the catalog is shown.

        Searching from the constructor would put a network call in the path of every GuPPy
        start, including runs that never touch DANDI, so the archive is not asked until the
        panel is actually looked at.
        """
        self.show_results()
        if self._searched:
            return
        self._searched = True
        self.refresh_catalog()

    def show_results(self, event: object = None) -> None:
        """Show the list of dandisets, leaving any dandiset that was open.

        Parameters
        ----------
        event : object, optional
            The Panel click event; unused.
        """
        self.dandiset_view.visible = False
        self.list_view.visible = True
        self.results_table.selection = []
        self._clear_selection()

    def _show_dandiset(self, summary: DandisetSummary) -> None:
        """Show one dandiset's page in place of the list."""
        self.dandiset_details.object = describe_dandiset(summary)
        self.list_view.visible = False
        self.dandiset_view.visible = True

    def _on_verify_listed(self, event: object) -> None:
        if not self.verify_listed.value or self._verification.get("running"):
            self.apply_filters()
            return
        candidates = list(self.visible_summaries)
        self._start_verification(
            references=order_for_verification([DandisetReference.from_summary(summary) for summary in candidates]),
            known_summaries={summary.identifier: summary for summary in candidates},
            description="listed dandiset",
        )

    def stop_verification(self, event: object = None) -> None:
        """Ask the running read to stop after the dandiset it is on.

        Parameters
        ----------
        event : object, optional
            The Panel click event; unused.
        """
        self._verification["stopping"] = True
        self.stop_button.disabled = True

    def _forget_verdicts(self) -> None:
        """Drop the previous search's verdicts and the option that produced them."""
        self.verdicts = {}
        self.verify_listed.value = False

    def _start_verification(
        self,
        *,
        references: list,
        known_summaries: dict,
        description: str,
    ) -> None:
        """Read ``references`` on a worker thread, recording each verdict as it settles.

        The work runs off the server IOLoop and is polled back onto it, because a run can
        last from seconds to an hour and blocking the loop for that would drop the browser's
        websocket. Confirmations arrive early and rejections late -- ruling a dandiset out
        means reading every asset it has -- so the answer fills in quickly and then slows.
        """
        total = len(references)
        if not total:
            self.status.object = "Nothing to read."
            return
        self.search_button.disabled = True
        self.stop_button.disabled = False
        self.stop_button.visible = True
        self.verification_progress.max = total
        self.verification_progress.value = 0
        self.verification_progress.visible = True

        self._verification = {
            "settled": 0,
            "total": total,
            "verdicts": {},
            "known": known_summaries,
            "description": description,
            "stopping": False,
            "running": True,
        }

        def on_verdict(reference: object, holds: bool | None) -> None:
            state = self._verification
            state["settled"] += 1
            state["verdicts"][reference.identifier] = holds

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
        """Draw the running read's progress and fold in the verdicts it has reached."""
        state = self._verification
        settled, total = state["settled"], state["total"]
        self.verification_progress.value = min(settled, total)
        verdicts = dict(state["verdicts"])
        confirmed = [identifier for identifier, holds in verdicts.items() if holds]
        finished = not state["thread"].is_alive()
        if finished or len(verdicts) != len(self.verdicts):
            self.verdicts = verdicts
            self.apply_filters()
        if finished:
            state["callback"].stop()
            state["running"] = False
            self.search_button.disabled = False
            self.stop_button.visible = False
            self.verification_progress.visible = False
        self.status.object = self._verification_status(
            settled=settled,
            total=total,
            confirmed=len(confirmed),
            unresolved=sum(1 for holds in verdicts.values() if holds is None),
            finished=finished,
        )

    def _verification_status(
        self,
        *,
        settled: int,
        total: int,
        confirmed: int,
        unresolved: int,
        finished: bool,
    ) -> str:
        """Describe how far the read has got and what its answer does not cover."""
        noun = self._verification["description"]
        holds = "holds" if confirmed == 1 else "hold"
        if not finished:
            return f"Read **{settled}** of {total} {noun}(s) so far — " f"**{confirmed}** {holds} fiber photometry."
        lead = "Stopped after" if self._verification["stopping"] else "Read"
        # An unread dandiset is not an empty one, and saying nothing about it would make the
        # two look alike.
        unreadable = (
            ""
            if not unresolved
            else (
                f" {unresolved} could not be read in full and are not accounted for either "
                "way; reading again retries them."
            )
        )
        return f"{lead} **{settled}** of {total} {noun}(s): " f"**{confirmed}** {holds} fiber photometry.{unreadable}"

    def apply_filters(self, event: object = None) -> None:
        """Redraw the table from the current results and whatever has been verified.

        Parameters
        ----------
        event : object, optional
            The Panel value-change event; unused.
        """
        listed = self.summaries
        if self.verify_listed.value and self.verdicts:
            listed = [summary for summary in listed if self.verdicts.get(summary.identifier)]
        self.visible_summaries = listed
        self.results_table.value = catalog_dataframe(self.visible_summaries)
        self.results_table.selection = []
        self._clear_selection()
        if self._verification.get("running"):
            return
        shown, total = len(self.visible_summaries), len(self.summaries)
        scope = f"**{shown}** dandiset(s)" if shown == total else f"**{shown}** of {total} dandiset(s)"
        self.status.object = f"{scope}. Select one to see its metadata."

    # ------------------------------------------------------------------------------------
    # Selection
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

    def _on_row_selected(self, event: object) -> None:
        summary = self.selected_summary
        if summary is None:
            return
        self._show_dandiset(summary)

    def _on_use_clicked(self, event: object = None) -> None:
        summary = self.selected_summary
        if summary is None:
            self.status.object = "⚠️ Select a dandiset first."
            return
        logger.info("DANDI browser: dandiset %s chosen for analysis", summary.identifier)
        if self.on_dandiset_selected is not None:
            self.on_dandiset_selected(summary.identifier)
