"""Panel component for browsing public DANDI dandisets and selecting NWB assets."""

import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from threading import Thread

import panel as pn
from dandi.exceptions import NotFoundError

from .dandi_browser import BROWSER_WIDTH, DandiBrowser, PhotometryPreviewPane
from .frontend_utils import default_root_path
from ..utils.dandi_catalog import (
    AssetSummary,
    filter_assets,
    format_byte_size,
    list_nwb_assets,
    preview_asset,
    scan_assets_for_photometry,
)

logger = logging.getLogger(__name__)

_DANDISET_ID_PATTERN = re.compile(r"^\d{6}$")

# Stable per-process parent directory under which we build a fake filesystem
# mirror of each dandiset (one subfolder per dandiset, containing zero-byte
# ``.nwb`` placeholders that follow the dandiset's real asset layout). This
# lets the user navigate DANDI assets with the same ``FileSelector`` they know
# from local mode. We leave cleanup to the OS — the parent lives under the
# system temp dir.
_MIRROR_ROOT = str(Path(tempfile.gettempdir()) / "guppy_dandi_mirror")


def _build_dandiset_mirror(*, dandiset_id: str, mirror_parent: str, assets: list[AssetSummary]) -> str:
    """Build a temp directory tree mirroring ``assets``, for the asset ``FileSelector`` to walk.

    For every asset path, create the intermediate directories and touch a zero-byte
    placeholder at the leaf. Any tree left from a previous load of the same dandiset is
    removed first, so narrowing the asset filters drops the placeholders they exclude rather
    than leaving them behind.

    Parameters
    ----------
    dandiset_id : str
        Six-digit dandiset ID, which names the tree's root directory.
    mirror_parent : str
        Parent directory the tree is materialized under.
    assets : list of AssetSummary
        The assets to mirror.

    Returns
    -------
    str
        Path to the dandiset's mirror root.
    """
    mirror_root = Path(mirror_parent) / dandiset_id
    shutil.rmtree(mirror_root, ignore_errors=True)
    mirror_root.mkdir(parents=True, exist_ok=True)
    for asset in assets:
        absolute_path = mirror_root / asset.path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        absolute_path.touch(exist_ok=True)
    return str(mirror_root)


class DandiSelector:
    """
    A Panel widget for finding and selecting NWB files from the public DANDI Archive.

    The panel walks four steps. A :class:`~guppy.frontend.dandi_browser.DandiBrowser` searches
    the archive's fiber photometry dandisets and hands the chosen identifier to the Dandiset ID
    field, which the user can also fill in by hand. Loading a dandiset lists its NWB assets,
    which two cheap filters — a path substring and a minimum file size — narrow before the
    listing is materialized as a temporary directory tree (zero-byte placeholders matching the
    asset layout) under the system temp dir. A ``pn.widgets.FileSelector`` then points at that
    tree, which matches the local-mode UX exactly: hierarchical navigation, click-to-descend,
    native multi-select. A selected asset can be streamed and previewed before it is analyzed.

    Selected absolute paths are translated back to ``dandi://`` URIs via ``selected_uris``.

    Parameters
    ----------
    styles : dict of {str: str} or None
        Panel styles applied to the composed layout.
    mirror_parent : str or None
        Parent directory the placeholder asset tree is materialized under.
    start_path : str or None
        Initial directory shown in the local output-directory selector. Falls back to
        ``default_root_path()`` when not supplied or when the path does not exist.
    list_assets_function : callable, optional
        Injection point for the asset listing; defaults to
        :func:`~guppy.utils.dandi_catalog.list_nwb_assets`.
    preview_function : callable, optional
        Injection point for the streaming preview; defaults to
        :func:`~guppy.utils.dandi_catalog.preview_asset`.
    scan_function : callable, optional
        Injection point for the fiber photometry scan; defaults to
        :func:`~guppy.utils.dandi_catalog.scan_assets_for_photometry`.
    browser : DandiBrowser or None, optional
        Catalog browser to embed. One is built when not supplied.

    Attributes
    ----------
    panel : panel.Column
        The composed Panel layout to embed in a template.
    selected_uris : list[str]
        Read-only property returning the currently-selected DANDI URIs in the
        form ``dandi://<dandiset_id>/<asset_path>``.
    output_root : str | None
        Read-only property returning the selected local output directory, or
        ``None`` if none is selected.
    """

    def __init__(
        self,
        *,
        styles: dict[str, str] | None = None,
        mirror_parent: str | None = None,
        start_path: str | None = None,
        list_assets_function: object = list_nwb_assets,
        preview_function: object = preview_asset,
        scan_function: object = scan_assets_for_photometry,
        browser: DandiBrowser | None = None,
    ) -> None:
        self.styles = styles or dict(background="WhiteSmoke")
        # Allow tests to inject a tmp_path-based parent; default to the
        # module-level stable location.
        self._mirror_parent = mirror_parent if mirror_parent is not None else _MIRROR_ROOT
        Path(self._mirror_parent).mkdir(parents=True, exist_ok=True)
        self.list_assets_function = list_assets_function
        self.preview_function = preview_function
        self.scan_function = scan_function

        self._current_mirror_root = None
        # The loaded dandiset's full NWB asset listing, which the asset filters narrow without
        # going back to the archive.
        self._assets: list[AssetSummary] = []
        # Asset path -> whether that asset holds fiber photometry, filled in by a scan.
        self._photometry_by_path: dict[str, bool] = {}
        # State of the scan currently running, if any: its counter, thread and poller.
        self._scan: dict[str, object] = {}
        # Re-attached to each rebuilt asset FileSelector by _make_asset_file_selector.
        self._asset_selection_watchers = []

        self.browser = browser if browser is not None else DandiBrowser(on_dandiset_selected=self.load_dandiset)
        self.browser_card = pn.Card(
            self.browser.panel,
            title="Find a fiber photometry dandiset",
            width=BROWSER_WIDTH + 30,
            collapsed=True,
        )

        self.dandiset_input = pn.widgets.TextInput(
            name="Dandiset ID",
            value="",
            placeholder="e.g. 000971",
            width=400,
        )
        self.dandiset_input.param.watch(self._on_dandiset_change, "value")

        self.scan_button = pn.widgets.Button(
            name="Scan for fiber photometry",
            button_type="primary",
            width=260,
            disabled=True,
            description=(
                "Reads the header of every listed NWB file straight from the archive and reports "
                "which of them hold fiber photometry. Nothing is downloaded; a few hundred "
                "kilobytes per file are read to answer the question."
            ),
        )
        self.scan_button.on_click(self.scan_assets)

        self.photometry_only = pn.widgets.Checkbox(
            name="Show only files with fiber photometry",
            value=False,
            disabled=True,
            width=320,
        )
        self.photometry_only.param.watch(self._on_asset_filter_change, "value")

        self.scan_progress = pn.indicators.Progress(
            name="Scanning",
            value=0,
            max=1,
            width=600,
            visible=False,
        )

        # Panel's FileSelector populates its listing once at construction and
        # does not re-scan when ``root_directory`` is reassigned programmatically.
        # To refresh the browser on each dandiset load, we rebuild the widget
        # and swap it into a stable slot in the layout.
        self.asset_file_selector = self._make_asset_file_selector(self._mirror_parent)
        self._asset_file_selector_slot = pn.Column(self.asset_file_selector)

        self.preview_button = pn.widgets.Button(name="Preview selected file", width=220)
        self.preview_button.on_click(self.preview_selected_asset)
        self.asset_preview_pane = PhotometryPreviewPane(preview_function=preview_function, width=BROWSER_WIDTH)

        self.output_root_selector = pn.widgets.FileSelector(
            start_path if start_path and Path(start_path).is_dir() else default_root_path(),
            root_directory="/",
            name="Local output directory",
            width=950,
        )

        self.status = pn.pane.Markdown("", width=950)
        self.asset_status = pn.pane.Markdown("", width=950)

        self.panel = pn.Column(
            pn.pane.Markdown(
                "### DANDI source\n"
                "Follow the steps below to stream one or more NWB files directly from the "
                "[DANDI Archive](https://dandiarchive.org) through the GuPPy pipeline."
            ),
            pn.pane.Markdown(
                "**Step 1:** Search the archive for a dataset to reanalyze, or skip ahead to "
                "Step 2 if you already know the Dandiset ID you want."
            ),
            self.browser_card,
            pn.pane.Markdown(
                "**Step 2:** Enter a public Dandiset ID below (six digits, e.g. `000971`). "
                "Its NWB assets will load automatically."
            ),
            self.dandiset_input,
            self.status,
            pn.pane.Markdown(
                "**Step 3:** Browse the dandiset's subject folders below and select one or more "
                "NWB files. Navigation works the same as local mode — click a folder to descend, "
                "Ctrl/Cmd-click to multi-select files. **Scan for fiber photometry** checks every "
                "listed file and lets you hide the ones that hold none, and **Preview selected "
                "file** streams the file's header to report which channels it holds before you "
                "commit to analyzing it."
            ),
            pn.Row(self.scan_button, self.photometry_only),
            self.scan_progress,
            self.asset_status,
            self._asset_file_selector_slot,
            self.preview_button,
            self.asset_preview_pane.panel,
            pn.pane.Markdown(
                "**Step 4:** Choose a local directory where pipeline outputs will be written. "
                "One subfolder will be created per selected asset."
            ),
            self.output_root_selector,
        )

    def _make_asset_file_selector(self, root_directory: str) -> pn.widgets.FileSelector:
        """Construct a fresh ``FileSelector`` rooted at ``root_directory``.

        Panel's ``FileSelector`` caches its listing at construction time, so we
        build a new widget on each dandiset change rather than mutating the
        existing one in place.
        """
        file_selector = pn.widgets.FileSelector(
            root_directory,
            root_directory=root_directory,
            file_pattern="*.nwb",
            name="NWB assets",
            width=950,
        )
        # Hide the mirror-path TextInput at the top of the FileSelector — users
        # should never see the internal /tmp/guppy_dandi_mirror/... path.
        file_selector._directory.visible = False
        for callback in self._asset_selection_watchers:
            file_selector.param.watch(callback, "value")
        return file_selector

    def attach_asset_selection_watcher(self, *, callback: object) -> None:
        """Call ``callback`` whenever the set of selected NWB assets changes.

        The asset ``FileSelector`` is rebuilt on every dandiset change, so watchers
        registered here are re-attached to each replacement and are also called
        directly on the swap, which drops the previous selection without firing a
        ``value`` event of its own.

        Parameters
        ----------
        callback : callable
            Receives the Panel ``value`` change event, or ``None`` on a rebuild.
        """
        self._asset_selection_watchers.append(callback)
        self.asset_file_selector.param.watch(callback, "value")

    def _swap_asset_file_selector(self, root_directory: str) -> None:
        self.asset_file_selector = self._make_asset_file_selector(root_directory)
        self._asset_file_selector_slot[:] = [self.asset_file_selector]
        for callback in self._asset_selection_watchers:
            callback(None)

    def _reset_to_empty(self) -> None:
        self._current_mirror_root = None
        self._assets = []
        self._forget_scan()
        self.asset_status.object = ""
        self.asset_preview_pane.clear()
        self._swap_asset_file_selector(self._mirror_parent)

    def _forget_scan(self) -> None:
        """Drop the previous dandiset's scan verdicts and disable the filter they fed."""
        self._photometry_by_path = {}
        self.photometry_only.value = False
        self.photometry_only.disabled = True
        self.scan_button.disabled = True

    def load_dandiset(self, dandiset_id: str) -> None:
        """Load ``dandiset_id`` into the Dandiset ID field, which lists its assets.

        This is what the catalog browser calls when the user picks a dandiset to analyze.

        Parameters
        ----------
        dandiset_id : str
            Six-digit dandiset ID.
        """
        self.dandiset_input.value = dandiset_id

    def _on_dandiset_change(self, event: object) -> None:
        dandiset_id = (event.new or "").strip()
        if not dandiset_id:
            self._reset_to_empty()
            self.status.object = ""
            return

        if not _DANDISET_ID_PATTERN.match(dandiset_id):
            self._reset_to_empty()
            self.status.object = f"⚠️ Invalid Dandiset ID `{dandiset_id}`. Expected exactly six digits, e.g. `000971`."
            return

        self.status.object = f"Fetching assets for Dandiset {dandiset_id}..."
        try:
            assets = self.list_assets_function(dandiset_id=dandiset_id)
        except NotFoundError:
            self._reset_to_empty()
            self.status.object = f"⚠️ Dandiset {dandiset_id} not found on the DANDI Archive."
            return

        self._assets = assets
        self._forget_scan()
        self.scan_button.disabled = not assets
        size_range = ""
        if assets:
            sizes = [asset.size_in_bytes for asset in assets]
            size_range = f" ranging {format_byte_size(min(sizes))} – {format_byte_size(max(sizes))}"
        self.status.object = f"✅ Dandiset {dandiset_id}: {len(assets)} NWB asset(s) found{size_range}."
        self._rebuild_mirror()

    def _on_asset_filter_change(self, event: object) -> None:
        if self._assets:
            self._rebuild_mirror()

    def _filtered_assets(self) -> list[AssetSummary]:
        """Return the loaded dandiset's assets that pass the fiber photometry filter."""
        return filter_assets(
            self._assets,
            photometry_by_path=self._photometry_by_path,
            photometry_only=self.photometry_only.value,
        )

    def scan_assets(self, event: object = None) -> None:
        """Scan every listed asset for fiber photometry and switch the filter on.

        The scan runs on a worker thread and is polled from the server IOLoop, rather than
        being waited on here: a synchronous wait would block the IOLoop for the whole scan,
        so the progress bar would never repaint and the browser tab would drop its
        websocket connection.
        """
        if not self._assets:
            return
        total = len(self._assets)
        self.scan_button.loading = True
        self.scan_button.disabled = True
        self.scan_progress.max = total
        self.scan_progress.value = 0
        self.scan_progress.visible = True
        self.asset_status.object = f"Scanning 0 of {total} NWB asset(s) for fiber photometry..."

        # ``completed`` is written only by the worker thread and read only by the poller, so
        # it needs no lock: the scan reports from the single thread it collects results on.
        self._scan = {"completed": 0, "verdicts": {}, "total": total}

        def worker() -> None:
            self._scan["verdicts"] = self.scan_function(self._assets, progress_callback=self._record_scan_progress)

        self._scan["thread"] = Thread(target=worker)
        self._scan["thread"].start()
        self._scan["callback"] = pn.state.add_periodic_callback(self._poll_scan, period=200)

    def _record_scan_progress(self, completed: int) -> None:
        """Note how many assets the running scan has finished, for the poller to render."""
        self._scan["completed"] = completed

    def _poll_scan(self) -> None:
        """Render the running scan's progress, and apply its verdicts once it finishes."""
        completed = self._scan["completed"]
        total = self._scan["total"]
        self.scan_progress.value = min(completed, total)
        self.asset_status.object = f"Scanning {completed} of {total} NWB asset(s) for fiber photometry..."
        # Completion is the worker thread finishing, never the count reaching the total, so
        # the verdicts are always fully assigned before they are read.
        if not self._scan["thread"].is_alive():
            self._scan["callback"].stop()
            self._finish_scan(self._scan["verdicts"])

    def _finish_scan(self, verdicts: dict[str, bool]) -> None:
        """Apply a finished scan's verdicts and turn the photometry filter on."""
        self._photometry_by_path = verdicts
        self.scan_progress.visible = False
        self.scan_button.loading = False
        self.scan_button.disabled = False
        self.photometry_only.disabled = False
        # Switching it on is why the button was pressed; the unscanned listing is one click away.
        self.photometry_only.value = True
        self._rebuild_mirror()

    def _rebuild_mirror(self) -> None:
        """Re-materialize the placeholder tree from the filtered assets and repoint the selector."""
        dandiset_id = (self.dandiset_input.value or "").strip()
        assets = self._filtered_assets()
        self._current_mirror_root = _build_dandiset_mirror(
            dandiset_id=dandiset_id, mirror_parent=self._mirror_parent, assets=assets
        )
        self.asset_preview_pane.clear()
        self._swap_asset_file_selector(self._current_mirror_root)
        if self._photometry_by_path:
            with_photometry = sum(1 for held in self._photometry_by_path.values() if held)
            scanned = (
                f" Scanned {len(self._photometry_by_path)} file(s): " f"**{with_photometry}** hold fiber photometry."
            )
        else:
            scanned = ""
        self.asset_status.object = (
            f"Showing **{len(assets)}** of {len(self._assets)} NWB asset(s) in the tree below.{scanned}"
        )

    def _selected_relative_paths(self) -> list[str]:
        if self._current_mirror_root is None:
            return []
        selected = []
        for absolute_path in self.asset_file_selector.value or []:
            if not str(absolute_path).endswith(".nwb"):
                continue
            relative = os.path.relpath(absolute_path, self._current_mirror_root)
            # Normalize to forward slashes for the DANDI URI regardless of OS.
            selected.append(relative.replace(os.sep, "/"))
        return selected

    def preview_selected_asset(self, event: object = None) -> None:
        """Stream the first selected asset's header and show what it holds.

        Parameters
        ----------
        event : object, optional
            The Panel click event; unused.
        """
        asset_paths = self._selected_relative_paths()
        if not asset_paths:
            self.asset_status.object = "⚠️ Select an NWB file in the tree above first."
            return
        dandiset_id = (self.dandiset_input.value or "").strip()
        asset_path = asset_paths[0]
        self.asset_status.object = f"Streaming `{asset_path}`..."
        preview = self.preview_function(dandiset_id=dandiset_id, asset_path=asset_path)
        self.asset_preview_pane.show(preview=preview)
        self.asset_status.object = f"Previewed `{asset_path}` from dandiset {dandiset_id}."

    @property
    def selected_uris(self) -> list[str]:
        """Return the currently selected DANDI URIs.

        Returns
        -------
        list of str
            Each element is a ``dandi://<dandiset_id>/<asset_path>`` URI for
            every ``.nwb`` placeholder currently selected in the asset browser.
            Returns an empty list when no dandiset has been loaded or no files
            are selected.
        """
        dandiset_id = (self.dandiset_input.value or "").strip()
        if not dandiset_id:
            return []
        return [f"dandi://{dandiset_id}/{path}" for path in self._selected_relative_paths()]

    @property
    def output_root(self) -> str | None:
        """Return the local output directory selected by the user.

        Returns
        -------
        str or None
            Absolute path of the first entry in the output-root
            ``FileSelector``'s value, or ``None`` when nothing is selected.
        """
        selected = self.output_root_selector.value
        if not selected:
            return None
        return selected[0]
