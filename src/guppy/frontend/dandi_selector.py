"""Panel component for browsing public DANDI dandisets and selecting NWB assets."""

import logging
import os
import tempfile
from pathlib import Path

import panel as pn
from dandi.dandiapi import DandiAPIClient

logger = logging.getLogger(__name__)

# Stable per-process parent directory under which we build a fake filesystem
# mirror of each dandiset (one subfolder per dandiset, containing zero-byte
# ``.nwb`` placeholders that follow the dandiset's real asset layout). This
# lets the user navigate DANDI assets with the same ``FileSelector`` they know
# from local mode. We leave cleanup to the OS — the parent lives under the
# system temp dir.
_MIRROR_ROOT = os.path.join(tempfile.gettempdir(), "guppy_dandi_mirror")


def _default_root_path():
    base_directory_environment = os.environ.get("GUPPY_BASE_DIR")
    if base_directory_environment and os.path.isdir(base_directory_environment):
        return base_directory_environment
    return os.path.expanduser("~")


def _build_dandiset_mirror(*, dandiset_id, mirror_parent):
    """Build (or reuse) a temp directory tree mirroring a dandiset's NWB assets.

    For every ``.nwb`` asset path returned by the DANDI API, create the
    intermediate directories and touch a zero-byte placeholder at the leaf.
    The resulting tree is walkable by ``pn.widgets.FileSelector``.

    Returns the path to the dandiset's mirror root.
    """
    mirror_root = os.path.join(mirror_parent, dandiset_id)
    os.makedirs(mirror_root, exist_ok=True)
    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(dandiset_id)
        asset_paths = [asset.path for asset in dandiset.get_assets() if asset.path.endswith(".nwb")]
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

        # Start pointed at the stable parent; will be repointed to
        # <parent>/<dandiset_id>/ on dandiset change. Restricted to .nwb files
        # so folders remain navigable but only NWB leaves are selectable.
        self.asset_file_selector = pn.widgets.FileSelector(
            self._mirror_parent,
            root_directory=self._mirror_parent,
            file_pattern="*.nwb",
            name="NWB assets",
            width=950,
        )
        # Hide the mirror-path TextInput at the top of the FileSelector — users
        # should never see the internal /tmp/guppy_dandi_mirror/... path.
        self.asset_file_selector._directory.visible = False

        self.output_root_selector = pn.widgets.FileSelector(
            _default_root_path(),
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
            self.asset_file_selector,
            pn.pane.Markdown(
                "**Step 3:** Choose a local directory where pipeline outputs will be written. "
                "One subfolder will be created per selected asset."
            ),
            self.output_root_selector,
        )

    def _on_dandiset_change(self, event):
        dandiset_id = (event.new or "").strip()
        self.asset_file_selector.value = []
        if not dandiset_id:
            self._current_mirror_root = None
            self.asset_file_selector.root_directory = self._mirror_parent
            self.asset_file_selector.directory = self._mirror_parent
            self.status.object = ""
            return

        self.status.object = f"Fetching assets for Dandiset {dandiset_id}..."
        mirror_root, asset_count = _build_dandiset_mirror(dandiset_id=dandiset_id, mirror_parent=self._mirror_parent)
        self._current_mirror_root = mirror_root
        self.asset_file_selector.root_directory = mirror_root
        self.asset_file_selector.directory = mirror_root
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
