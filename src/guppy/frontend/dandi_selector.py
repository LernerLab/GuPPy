"""Panel component for browsing public DANDI dandisets and selecting NWB assets."""

import logging
import os
import re
import tempfile
from pathlib import Path

import panel as pn
from dandi.dandiapi import DandiAPIClient
from dandi.exceptions import NotFoundError

from .frontend_utils import default_root_path

logger = logging.getLogger(__name__)

_DANDISET_ID_PATTERN = re.compile(r"^\d{6}$")

# Stable per-process parent directory under which we build a fake filesystem
# mirror of each dandiset (one subfolder per dandiset, containing zero-byte
# ``.nwb`` placeholders that follow the dandiset's real asset layout). This
# lets the user navigate DANDI assets with the same ``FileSelector`` they know
# from local mode. We leave cleanup to the OS — the parent lives under the
# system temp dir.
_MIRROR_ROOT = os.path.join(tempfile.gettempdir(), "guppy_dandi_mirror")


def _build_dandiset_mirror(*, dandiset_id, mirror_parent):
    """Build (or reuse) a temp directory tree mirroring a dandiset's NWB assets.

    For every ``.nwb`` asset path returned by the DANDI API, create the
    intermediate directories and touch a zero-byte placeholder at the leaf.
    The resulting tree is walkable by ``pn.widgets.FileSelector``.

    Returns the path to the dandiset's mirror root.
    """
    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(dandiset_id)
        asset_paths = [asset.path for asset in dandiset.get_assets() if asset.path.endswith(".nwb")]
    mirror_root = os.path.join(mirror_parent, dandiset_id)
    os.makedirs(mirror_root, exist_ok=True)
    for asset_path in asset_paths:
        absolute_path = os.path.join(mirror_root, asset_path)
        os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
        Path(absolute_path).touch(exist_ok=True)
    return mirror_root, len(asset_paths)


class DandiSelector:
    """
    A Panel widget for selecting NWB files from the public DANDI Archive.

    The user types a public Dandiset ID; the component fetches its asset list
    and materializes it as a temporary directory tree (zero-byte placeholders
    matching the asset layout) under the system temp dir, then points a
    ``pn.widgets.FileSelector`` at that tree. This matches the local-mode
    ``FileSelector`` UX exactly — hierarchical navigation, click-to-descend,
    native multi-select.

    Selected absolute paths are translated back to ``dandi://`` URIs via
    ``selected_uris``.

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

    def __init__(self, *, styles=None, mirror_parent=None):
        self.styles = styles or dict(background="WhiteSmoke")
        # Allow tests to inject a tmp_path-based parent; default to the
        # module-level stable location.
        self._mirror_parent = mirror_parent if mirror_parent is not None else _MIRROR_ROOT
        os.makedirs(self._mirror_parent, exist_ok=True)

        self._current_mirror_root = None

        self.dandiset_input = pn.widgets.TextInput(
            name="Dandiset ID",
            value="",
            placeholder="e.g. 000971",
            width=400,
        )
        self.dandiset_input.param.watch(self._on_dandiset_change, "value")

        # Panel's FileSelector populates its listing once at construction and
        # does not re-scan when ``root_directory`` is reassigned programmatically.
        # To refresh the browser on each dandiset load, we rebuild the widget
        # and swap it into a stable slot in the layout.
        self.asset_file_selector = self._make_asset_file_selector(self._mirror_parent)
        self._asset_file_selector_slot = pn.Column(self.asset_file_selector)

        self.output_root_selector = pn.widgets.FileSelector(
            default_root_path(),
            root_directory="/",
            name="Local output directory",
            width=950,
        )

        self.status = pn.pane.Markdown("", width=950)

        self.panel = pn.Column(
            pn.pane.Markdown(
                "### DANDI source\n"
                "Follow the steps below to stream one or more NWB files directly from the "
                "[DANDI Archive](https://dandiarchive.org) through the GuPPy pipeline."
            ),
            pn.pane.Markdown(
                "**Step 1:** Enter a public Dandiset ID below (six digits, e.g. `000971`). "
                "Its NWB assets will load automatically."
            ),
            self.dandiset_input,
            self.status,
            pn.pane.Markdown(
                "**Step 2:** Browse the dandiset's subject folders below and select one or more "
                "NWB files. Navigation works the same as local mode — click a folder to descend, "
                "Ctrl/Cmd-click to multi-select files."
            ),
            self._asset_file_selector_slot,
            pn.pane.Markdown(
                "**Step 3:** Choose a local directory where pipeline outputs will be written. "
                "One subfolder will be created per selected asset."
            ),
            self.output_root_selector,
        )

    def _make_asset_file_selector(self, root_directory):
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
        return file_selector

    def _swap_asset_file_selector(self, root_directory):
        self.asset_file_selector = self._make_asset_file_selector(root_directory)
        self._asset_file_selector_slot[:] = [self.asset_file_selector]

    def _reset_to_empty(self):
        self._current_mirror_root = None
        self._swap_asset_file_selector(self._mirror_parent)

    def _on_dandiset_change(self, event):
        dandiset_id = (event.new or "").strip()
        if not dandiset_id:
            self._reset_to_empty()
            self.status.object = ""
            return

        if not _DANDISET_ID_PATTERN.match(dandiset_id):
            self._reset_to_empty()
            self.status.object = (
                f"\u26a0\ufe0f Invalid Dandiset ID `{dandiset_id}`. Expected exactly six digits, e.g. `000971`."
            )
            return

        self.status.object = f"Fetching assets for Dandiset {dandiset_id}..."
        try:
            mirror_root, asset_count = _build_dandiset_mirror(
                dandiset_id=dandiset_id, mirror_parent=self._mirror_parent
            )
        except NotFoundError:
            self._reset_to_empty()
            self.status.object = f"\u26a0\ufe0f Dandiset {dandiset_id} not found on the DANDI Archive."
            return

        self._current_mirror_root = mirror_root
        self._swap_asset_file_selector(mirror_root)
        self.status.object = f"\u2705 Dandiset {dandiset_id}: {asset_count} NWB asset(s) loaded."

    def _selected_relative_paths(self):
        if self._current_mirror_root is None:
            return []
        selected = []
        for absolute_path in self.asset_file_selector.value or []:
            if not absolute_path.endswith(".nwb"):
                continue
            relative = os.path.relpath(absolute_path, self._current_mirror_root)
            # Normalize to forward slashes for the DANDI URI regardless of OS.
            selected.append(relative.replace(os.sep, "/"))
        return selected

    @property
    def selected_uris(self):
        dandiset_id = (self.dandiset_input.value or "").strip()
        if not dandiset_id:
            return []
        return [f"dandi://{dandiset_id}/{path}" for path in self._selected_relative_paths()]

    @property
    def output_root(self):
        selected = self.output_root_selector.value
        if not selected:
            return None
        return selected[0]
