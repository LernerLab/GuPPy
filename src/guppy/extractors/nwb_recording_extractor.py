"""NWB recording extractor for GuPPy fiber photometry pipeline."""

import glob
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from pynwb import NWBHDF5IO

from guppy.extractors.base_recording_extractor import BaseRecordingExtractor

logger = logging.getLogger(__name__)

# HDF5 dataset paths (relative to file root) that contain time-series data and
# should be truncated to num_samples when stubbing.
_ACQUISITION_DATA_SUFFIXES = ("/data", "/timestamps")


class NwbRecordingExtractor(BaseRecordingExtractor):
    """
    Recording extractor for NWB (Neurodata Without Borders) fiber photometry files.

    Reads acquisition series from ``*.nwb`` files found in a session folder.
    Multi-channel series (2-D data arrays) produce one event per column, named
    ``{acquisition_key}_{column_index}``.  Single-channel series produce one
    event named ``{acquisition_key}``.
    """

    @classmethod
    def discover_events_and_flags(cls, *, folder_path) -> tuple[list[str], list[str]]:
        """
        Discover available events from an NWB file.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing a single ``*.nwb`` file.

        Returns
        -------
        events : list of str
            Event names derived from acquisition series in the NWB file.
            Multi-channel series yield ``{key}_0``, ``{key}_1``, … per column.
            Single-channel series yield ``{key}``.
        flags : list of str
            Always empty for NWB files.
        """
        nwb_path = _find_nwb_file(folder_path)
        events = []
        with NWBHDF5IO(nwb_path, "r") as io:
            nwbfile = io.read()
            for key, series in nwbfile.acquisition.items():
                data_shape = series.data.shape
                if len(data_shape) == 2:
                    for column_index in range(data_shape[1]):
                        events.append(f"{key}_{column_index}")
                else:
                    events.append(key)

        return events, []

    def __init__(self, *, folder_path):
        self.folder_path = folder_path

    def read(self, *, events: list[str], outputPath: str) -> list[dict[str, Any]]:
        """
        Read data for the given events from the NWB file.

        Parameters
        ----------
        events : list of str
            Event names to extract (as returned by ``discover_events_and_flags``).
        outputPath : str
            Path to the output directory (unused during read; required by interface).

        Returns
        -------
        list of dict
            One dictionary per event with keys: ``storename``, ``sampling_rate``,
            ``timestamps``, ``data``, ``npoints``, ``channels``.
        """
        nwb_path = _find_nwb_file(self.folder_path)
        output_dicts = []

        with NWBHDF5IO(nwb_path, "r") as io:
            nwbfile = io.read()
            for event in events:
                acquisition_key, column_index = _parse_event_name(event, nwbfile)
                series = nwbfile.acquisition[acquisition_key]
                full_data = series.data[:]

                sampling_rate, timestamps = _resolve_timing(series, full_data.shape[0])

                if column_index is not None:
                    channel_data = full_data[:, column_index]
                else:
                    channel_data = full_data

                output_dicts.append(
                    {
                        "storename": event,
                        "sampling_rate": sampling_rate,
                        "timestamps": timestamps,
                        "data": channel_data,
                        "npoints": 1,
                    }
                )

        return output_dicts

    def save(self, *, output_dicts: list[dict[str, Any]], outputPath: str) -> None:
        """
        Save extracted data dictionaries to HDF5 format.

        Parameters
        ----------
        output_dicts : list of dict
            Data dictionaries returned by ``read()``.
        outputPath : str
            Directory where one ``.hdf5`` file per event will be written.
        """
        for S in output_dicts:
            event = S["storename"]
            self._write_hdf5(S["storename"], event, outputPath, "storename")
            self._write_hdf5(S["sampling_rate"], event, outputPath, "sampling_rate")
            self._write_hdf5(S["timestamps"], event, outputPath, "timestamps")
            self._write_hdf5(S["data"], event, outputPath, "data")
            self._write_hdf5(S["npoints"], event, outputPath, "npoints")

    def stub(self, *, folder_path, duration_in_seconds=1.0):
        """
        Create a truncated copy of the NWB file retaining only a short duration.

        Copies the source NWB file to ``folder_path``, truncating all acquisition
        data and timestamp arrays to the first ``duration_in_seconds`` of data.
        The sampling rate is determined from the first acquisition series found.

        Parameters
        ----------
        folder_path : str or Path
            Destination directory for the stubbed data.
        duration_in_seconds : float, optional
            Duration of data to retain. Default is 1.0.
        """
        src_nwb_path = _find_nwb_file(self.folder_path)

        dest_folder = Path(folder_path)
        if dest_folder.exists():
            shutil.rmtree(dest_folder)
        dest_folder.mkdir(parents=True)

        dest_nwb_path = dest_folder / Path(src_nwb_path).name

        # Determine num_samples from the first acquisition series
        with NWBHDF5IO(src_nwb_path, "r") as io:
            nwbfile = io.read()
            first_series = next(iter(nwbfile.acquisition.values()))
            sampling_rate, _ = _resolve_timing(first_series, first_series.data.shape[0])

        num_samples = int(sampling_rate * duration_in_seconds)

        # Copy HDF5 structure, truncating acquisition data/timestamp datasets
        with h5py.File(src_nwb_path, "r") as source_file:
            with h5py.File(str(dest_nwb_path), "w") as dest_file:
                _copy_hdf5_truncated(source_file, dest_file, num_samples, current_path="/")


def _find_nwb_file(folder_path):
    """Return the single NWB file path in *folder_path*, raising if not exactly one."""
    nwb_paths = glob.glob(os.path.join(folder_path, "*.nwb"))
    if len(nwb_paths) > 1:
        raise Exception("Two NWB files are present at the location.")
    if len(nwb_paths) == 0:
        raise Exception(f"No NWB file found in {folder_path}.")
    return nwb_paths[0]


def _parse_event_name(event, nwbfile):
    """
    Resolve an event name to an acquisition key and optional column index.

    Parameters
    ----------
    event : str
        Event name as produced by ``discover_events_and_flags``.
    nwbfile : NWBFile
        Open NWB file object to look up acquisition keys.

    Returns
    -------
    acquisition_key : str
    column_index : int or None
        ``None`` for single-channel (1-D) series.
    """
    if event in nwbfile.acquisition:
        return event, None

    # Parse "{acquisition_key}_{column_index}" format
    parts = event.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        key, column_index = parts[0], int(parts[1])
        if key in nwbfile.acquisition:
            return key, column_index

    raise ValueError(f"Event '{event}' could not be resolved to any acquisition series in the NWB file.")


def _resolve_timing(series, num_samples):
    """
    Derive sampling rate and a full timestamps array from an NWB TimeSeries.

    Checks ``rate``, then ``timestamps``, then reconstructs from ``starting_time``
    and ``rate`` (or raises if neither is available).

    Parameters
    ----------
    series : pynwb.TimeSeries
        The acquisition series to inspect.
    num_samples : int
        Total number of samples in the series data array.

    Returns
    -------
    sampling_rate : float
    timestamps : np.ndarray, shape (num_samples,)
    """
    sampling_rate = getattr(series, "rate", None)
    timestamps = getattr(series, "timestamps", None)

    if timestamps is not None:
        timestamps = np.array(timestamps[:])
        if sampling_rate is None:
            sampling_rate = 1.0 / float(np.median(np.diff(timestamps)))
    elif sampling_rate is not None:
        starting_time = getattr(series, "starting_time", None) or 0.0
        timestamps = starting_time + np.arange(num_samples) / sampling_rate
    else:
        raise Exception(f"Series '{series.name}' must have either 'rate' or 'timestamps' to reconstruct timing.")

    return float(sampling_rate), timestamps


def _copy_hdf5_truncated(source_group, dest_group, num_samples, current_path):
    """
    Recursively copy an HDF5 group, truncating acquisition time-series datasets.

    Datasets whose HDF5 path ends with ``/data`` or ``/timestamps`` and whose
    first dimension exceeds ``num_samples`` are sliced to ``[:num_samples]``.
    All other datasets and groups are copied verbatim.

    Parameters
    ----------
    source_group : h5py.Group
        Source HDF5 group (or file root).
    dest_group : h5py.Group
        Destination HDF5 group (or file root).
    num_samples : int
        Maximum number of samples to retain in truncated datasets.
    current_path : str
        HDF5 path of ``source_group`` relative to the file root, used to
        determine whether a dataset lives under ``/acquisition/``.
    """
    # Copy group-level attributes
    for attr_key, attr_value in source_group.attrs.items():
        dest_group.attrs[attr_key] = attr_value

    for key, item in source_group.items():
        item_path = current_path + key if current_path == "/" else current_path + "/" + key

        if isinstance(item, h5py.Group):
            new_group = dest_group.create_group(key)
            _copy_hdf5_truncated(item, new_group, num_samples, item_path)

        elif isinstance(item, h5py.Dataset):
            should_truncate = (
                item_path.startswith("/acquisition/")
                and any(item_path.endswith(suffix) for suffix in _ACQUISITION_DATA_SUFFIXES)
                and item.ndim >= 1
                and item.shape[0] > num_samples
            )

            if should_truncate:
                data = item[:num_samples]
            else:
                data = item[()]

            dest_group.create_dataset(key, data=data)
            for attr_key, attr_value in item.attrs.items():
                dest_group[key].attrs[attr_key] = attr_value
