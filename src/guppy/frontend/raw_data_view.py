"""Read-only view of the raw data Step 2 (Read Raw Data) wrote into each run folder.

Composed by ``orchestration/raw_data_view.py`` and served as a page on the main app once
the step finishes.
"""

from pathlib import Path

import h5py
import holoviews as hv
import numpy as np
import panel as pn

from .artifact_removal import build_run_folder_page
from .plot_card import CHECKBOX_MARGIN, CONTROL_MARGIN, plot_page, toolbar_field
from ..analysis.io_utils import (
    is_channel_label,
    is_covariate_label,
    recording_site_from_channel_label,
)
from ..analysis.timestamp_correction import expand_block_timestamps
from ..utils.stores_list import read_stores_list
from ..visualization.labels import trace_title
from ..visualization.raw_data import build_raw_traces, event_colors

hv.extension("bokeh")


def _raw_store_path(*, run_folder: str, store_id: str) -> Path:
    # Step 2 names each file after its store id, with path separators replaced.
    return Path(run_folder) / (store_id.replace("\\", "_").replace("/", "_") + ".hdf5")


def load_raw_data(
    *, run_folder: str
) -> tuple[dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], dict[str, np.ndarray]]:
    """Load the control and signal channels and the event times Step 2 wrote into a run folder.

    Times are in seconds from the run's earliest channel sample, so formats that record on an
    absolute clock read the same as those that start at zero. Event times are shifted by the
    same amount and so keep their place against the traces.

    Parameters
    ----------
    run_folder : str
        Run directory holding ``storesList.csv`` and the per-store HDF5 files.

    Returns
    -------
    channels : dict
        Recording site → channel label → ``(times, values)``, one time per sample. Sites are
        sorted by name and the control channel comes before the signal.
    event_times : dict
        Event label → event times, sorted by label. Covariates are left out.
    """
    store_array = read_stores_list(run_folder=run_folder)
    stores = sorted(zip(store_array[0], store_array[1], strict=True), key=lambda pair: pair[1])

    channels: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    event_times: dict[str, np.ndarray] = {}
    for store_id, store_label in stores:
        if is_covariate_label(store_label):
            continue
        with h5py.File(_raw_store_path(run_folder=run_folder, store_id=store_id), "r") as hdf5_file:
            times = np.asarray(hdf5_file["timestamps"]).ravel()
            if not is_channel_label(store_label):
                event_times[store_label] = times
                continue
            values = np.asarray(hdf5_file["data"]).ravel()
            if "npoints" in hdf5_file:
                times = expand_block_timestamps(
                    block_timestamps=times,
                    npoints=int(np.asarray(hdf5_file["npoints"])),
                    sampling_rate=float(np.asarray(hdf5_file["sampling_rate"])),
                )
        sample_count = min(times.shape[0], values.shape[0])
        site = recording_site_from_channel_label(store_label)
        channels.setdefault(site, {})[store_label] = (times[:sample_count], values[:sample_count])

    origin = min(times[0] for site_channels in channels.values() for times, _ in site_channels.values())
    channels = {
        site: {label: (times - origin, values) for label, (times, values) in site_channels.items()}
        for site, site_channels in sorted(channels.items(), key=lambda item: item[0].casefold())
    }
    event_times = {label: times - origin for label, times in event_times.items()}
    return channels, event_times


class RawDataView:
    """Read-only view of one run folder's raw data, with a recording-site selector.

    A recording site's control and signal channels stack in one axis-linked layout, with
    the times of the chosen events drawn over them.

    Parameters
    ----------
    channels : dict
        Recording site → channel label → ``(times, values)``.
    event_times : dict
        Event label → event timestamps.
    run_field : pn.viewable.Viewable
        The "Run Folder" toolbar field, placed first in the page's toolbar.
    """

    def __init__(
        self,
        *,
        channels: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
        event_times: dict[str, np.ndarray],
        run_field: pn.viewable.Viewable,
    ) -> None:
        self.channels = channels
        self.event_times = event_times
        self.colors = event_colors(event_labels=list(event_times))

        sites = list(channels)
        self.site_select = pn.widgets.Select(options=sites, value=sites[0], width=220, margin=CONTROL_MARGIN)
        self.event_select = pn.widgets.CheckBoxGroup(
            options=list(event_times), value=list(event_times), inline=True, margin=CHECKBOX_MARGIN
        )
        self.plot_pane = pn.pane.HoloViews(self._make_plot(), sizing_mode="stretch_width")
        self.site_select.param.watch(self._refresh, "value")
        self.event_select.param.watch(self._refresh, "value")

        fields = [run_field, toolbar_field("Recording Site", self.site_select)]
        if event_times:
            fields.append(toolbar_field("Events", self.event_select))
        self.widget = plot_page(fields=fields, body=[self.plot_pane])

    def _make_plot(self) -> hv.Layout:
        return build_raw_traces(
            traces={trace_title(channel): trace for channel, trace in self.channels[self.site_select.value].items()},
            event_times={label: self.event_times[label] for label in self.event_select.value},
            colors=self.colors,
        )

    def _refresh(self, event: object) -> None:
        self.plot_pane.object = self._make_plot()


def _build_folder_page(run_folder: str, run_field: pn.viewable.Viewable) -> pn.viewable.Viewable:
    channels, event_times = load_raw_data(run_folder=run_folder)
    return RawDataView(channels=channels, event_times=event_times, run_field=run_field).widget


def build_raw_data_view_page(*, run_folders: list[str]) -> pn.viewable.Viewable:
    """Compose the read-only Step-2 raw data view across all run folders.

    Parameters
    ----------
    run_folders : list of str
        Run directories the Read Raw Data step wrote into.

    Returns
    -------
    pn.viewable.Viewable
        The composed page.
    """
    return build_run_folder_page(run_folders=run_folders, build_folder_page=_build_folder_page)
