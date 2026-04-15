"""Panel component for browsing public DANDI dandisets and selecting NWB assets."""

import logging
import os

import panel as pn
from dandi.dandiapi import DandiAPIClient

logger = logging.getLogger(__name__)


def _default_root_path():
    base_directory_environment = os.environ.get("GUPPY_BASE_DIR")
    if base_directory_environment and os.path.isdir(base_directory_environment):
        return base_directory_environment
    return os.path.expanduser("~")


class DandiSelector:
    """
    A Panel widget for selecting NWB files from the public DANDI Archive.

    Exposes three interactive widgets: a dandiset chooser populated on demand
    via ``DandiAPIClient().get_dandisets()``; a multi-select of ``.nwb`` assets
    under the chosen dandiset; and a local FileSelector for the output root
    where pipeline outputs will be written.

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

    def __init__(self, *, styles=None):
        self.styles = styles or dict(background="WhiteSmoke")

        self.load_button = pn.widgets.Button(name="Load Public Dandisets", button_type="primary", width=220)
        self.load_button.on_click(self._on_load_dandisets)

        self.dandiset_input = pn.widgets.AutocompleteInput(
            name="Dandiset ID",
            options=[],
            case_sensitive=False,
            search_strategy="includes",
            placeholder="Click 'Load Public Dandisets' to populate",
            width=400,
            restrict=True,
        )
        self.dandiset_input.param.watch(self._on_dandiset_change, "value")

        self.asset_select = pn.widgets.MultiSelect(
            name="NWB assets (Ctrl/Cmd-click to multi-select)",
            options=[],
            size=12,
            width=950,
        )

        self.output_root_selector = pn.widgets.FileSelector(
            _default_root_path(),
            root_directory="/",
            name="Local output directory",
            width=950,
        )

        self.status = pn.pane.Markdown("", width=950)

        self.panel = pn.Column(
            pn.pane.Markdown(
                "**DANDI source:** pick a dandiset, then one or more NWB files. "
                "Pipeline outputs will be written under the local output directory below."
            ),
            self.load_button,
            self.dandiset_input,
            self.asset_select,
            pn.pane.Markdown("**Local output directory (one folder will be created per selected asset):**"),
            self.output_root_selector,
            self.status,
        )

    def _on_load_dandisets(self, event):
        with DandiAPIClient() as client:
            dandisets = list(client.get_dandisets())
        identifiers = sorted(dandiset.identifier for dandiset in dandisets)
        self.dandiset_input.options = identifiers
        self.status.object = f"Loaded {len(identifiers)} public dandisets."

    def _on_dandiset_change(self, event):
        dandiset_id = event.new
        if not dandiset_id:
            self.asset_select.options = []
            return
        with DandiAPIClient() as client:
            dandiset = client.get_dandiset(dandiset_id)
            asset_paths = [asset.path for asset in dandiset.get_assets() if asset.path.endswith(".nwb")]
        self.asset_select.options = sorted(asset_paths)
        self.status.object = f"Dandiset {dandiset_id}: {len(asset_paths)} NWB asset(s)."

    @property
    def selected_uris(self):
        dandiset_id = self.dandiset_input.value
        if not dandiset_id:
            return []
        return [f"dandi://{dandiset_id}/{asset_path}" for asset_path in self.asset_select.value]

    @property
    def output_root(self):
        selected = self.output_root_selector.value
        if not selected:
            return None
        return selected[0]
