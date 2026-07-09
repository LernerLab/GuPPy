"""NWB recording extractor for GuPPy fiber photometry pipeline."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from hdmf.common import DynamicTable
from pynwb import NWBHDF5IO, NWBFile

from guppy.extractors.base_recording_extractor import BaseRecordingExtractor
from guppy.utils._hdf5_io import write_hdf5

logger = logging.getLogger(__name__)


class NwbRecordingExtractor(BaseRecordingExtractor):
    """
    Recording extractor for NWB (Neurodata Without Borders) fiber photometry files.

    Discovers all ``FiberPhotometryResponseSeries`` objects anywhere in the NWB
    file (not limited to ``/acquisition``).  Multi-channel series (2-D data
    arrays) produce one event per column, named ``{series_name}_{column_index}``.
    Single-channel series produce one event named ``{series_name}``.
    """

    @classmethod
    def discover_events_and_flags(cls, folder_path: str | Path) -> tuple[list[str], list[str]]:
        """
        Discover available events from an NWB file.

        Parameters
        ----------
        folder_path : str or Path
            Path to the folder containing a single ``*.nwb`` file.

        Returns
        -------
        events : list of str
            Event names derived from ``FiberPhotometryResponseSeries`` objects.
            Multi-channel series yield ``{name}_0``, ``{name}_1``, … per column.
            Single-channel series yield ``{name}``.
        flags : list of str
            Always empty for NWB files.
        """
        nwb_path = _find_nwb_file(folder_path)

        with NWBHDF5IO(nwb_path, "r") as io:
            nwbfile = io.read()
            events = _discover_events_from_nwbfile(nwbfile=nwbfile, io=io)

        return events, []

    def __init__(self, *, folder_path: str | Path) -> None:
        self.folder_path = folder_path

    def count_samples(self, *, event: str) -> int:
        """Return the total number of samples for ``event`` without reading bulk data."""
        nwb_path = _find_nwb_file(self.folder_path)
        with NWBHDF5IO(nwb_path, "r") as io:
            nwbfile = io.read()
            return _count_event_samples(nwbfile=nwbfile, io=io, event=event)

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
            ``timestamps``, ``data``, ``npoints``.
        """
        nwb_path = _find_nwb_file(self.folder_path)

        with NWBHDF5IO(nwb_path, "r") as io:
            nwbfile = io.read()
            output_dicts = _read_events_from_nwbfile(nwbfile=nwbfile, io=io, events=events)

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
        for output_dict in output_dicts:
            event = output_dict["storename"]
            write_hdf5(output_dict["storename"], event, outputPath, "storename")
            write_hdf5(output_dict["timestamps"], event, outputPath, "timestamps")
            if "sampling_rate" in output_dict:
                write_hdf5(output_dict["sampling_rate"], event, outputPath, "sampling_rate")
            if "data" in output_dict:
                write_hdf5(output_dict["data"], event, outputPath, "data")
            if "npoints" in output_dict:
                write_hdf5(output_dict["npoints"], event, outputPath, "npoints")

    def stub(self, *, folder_path: str | Path, duration_in_seconds: float = 1.0) -> None:
        """
        Stub method is unnecessary for NWB files.
        """
        raise NotImplementedError("Stub method is unnecessary for NWB files.")


def _count_event_samples(*, nwbfile: NWBFile, io: NWBHDF5IO, event: str) -> int:
    """Return total samples for ``event`` without materialising bulk data.

    For ``FiberPhotometryResponseSeries`` events the result is the dataset's
    leading dimension. For ndx-events and core ``EventsTable`` events the result
    is ``0`` because they carry only sparse event timestamps and contribute
    negligibly to progress-bar weighting.
    """
    series_name_to_object = {
        neurodata_object.name: neurodata_object
        for neurodata_object in nwbfile.objects.values()
        if getattr(neurodata_object, "neurodata_type", None) == "FiberPhotometryResponseSeries"
    }
    if event in series_name_to_object:
        return int(series_name_to_object[event].data.shape[0])
    parts = event.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        series_name = parts[0]
        if series_name in series_name_to_object:
            return int(series_name_to_object[series_name].data.shape[0])
    return 0


def _discover_events_from_nwbfile(*, nwbfile: NWBFile, io: NWBHDF5IO) -> list[str]:
    """
    Extract event names from an already-open NWB file.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        An open NWB file object.
    io : pynwb.NWBHDF5IO
        The IO instance used to read the file (needed for namespace inspection).

    Returns
    -------
    events : list of str
        Event names derived from ``FiberPhotometryResponseSeries`` and ndx-events objects.
    """
    events = []
    seen_names = set()
    ndx_events_version = _get_ndx_events_version(io)

    for neurodata_object in nwbfile.objects.values():
        if getattr(neurodata_object, "neurodata_type", None) != "FiberPhotometryResponseSeries":
            continue
        series_name = neurodata_object.name
        _register_unique_name(seen_names, series_name, "FiberPhotometryResponseSeries")

        data_shape = neurodata_object.data.shape
        if len(data_shape) == 2:
            for column_index in range(data_shape[1]):
                events.append(f"{series_name}_{column_index}")
        else:
            events.append(series_name)

    if ndx_events_version is not None and ndx_events_version.startswith("0.2"):
        events.extend(_discover_ndx_events_v02(nwbfile, seen_names))

    events.extend(_discover_core_events_tables(nwbfile, seen_names))

    return events


def _read_events_from_nwbfile(*, nwbfile: NWBFile, io: NWBHDF5IO, events: list[str]) -> list[dict[str, Any]]:
    """
    Read data for specified events from an already-open NWB file.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        An open NWB file object.
    io : pynwb.NWBHDF5IO
        The IO instance used to read the file (needed for namespace inspection).
    events : list of str
        Event names to extract.

    Returns
    -------
    list of dict
        One dictionary per event with keys: ``storename``, ``sampling_rate``,
        ``timestamps``, ``data``, ``npoints``.
    """
    ndx_events_version = _get_ndx_events_version(io)

    series_name_to_object = {
        neurodata_object.name: neurodata_object
        for neurodata_object in nwbfile.objects.values()
        if getattr(neurodata_object, "neurodata_type", None) == "FiberPhotometryResponseSeries"
    }

    event_index = {}
    if ndx_events_version is not None and ndx_events_version.startswith("0.2"):
        event_index = _build_event_index_v02(nwbfile)
    event_index.update(_build_core_events_index(nwbfile))

    output_dicts = []
    for event in events:
        if event in event_index:
            output_dicts.append(_read_ndx_event(event_name=event, source_info=event_index[event]))
            continue

        series_name, column_index = _parse_event_name(event, series_name_to_object)
        series = series_name_to_object[series_name]
        full_data = series.data[:]
        sampling_rate, timestamps = _resolve_timing(series, full_data.shape[0])
        channel_data = full_data[:, column_index] if column_index is not None else full_data

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


def _register_unique_name(seen_names: set[str], name: str, type_label: str) -> None:
    """Add *name* to *seen_names*, raising ``ValueError`` if already present."""
    if name in seen_names:
        raise ValueError(
            f"Duplicate {type_label} name found: {name!r}. NWB object names must be unique within "
            "the file; rename the duplicate in the source NWB file before re-running."
        )
    seen_names.add(name)


def _discover_ndx_events_v02(nwbfile: NWBFile, seen_names: set[str]) -> list[str]:
    """Discover event names from ndx-events v0.2 objects (AnnotatedEventsTable, LabeledEvents, Events)."""
    events = []
    for neurodata_object in nwbfile.objects.values():
        neurodata_type = getattr(neurodata_object, "neurodata_type", None)
        if neurodata_type == "AnnotatedEventsTable":
            _register_unique_name(seen_names, neurodata_object.name, "AnnotatedEventsTable")
            for label in neurodata_object["label"].data:
                events.append(f"{neurodata_object.name}_{label}")
        elif neurodata_type == "LabeledEvents":
            _register_unique_name(seen_names, neurodata_object.name, "LabeledEvents")
            for label in neurodata_object.data__labels:
                events.append(f"{neurodata_object.name}_{label}")
        elif neurodata_type == "Events":
            _register_unique_name(seen_names, neurodata_object.name, "Events")
            events.append(neurodata_object.name)
    return events


def _core_events_split_column(table: DynamicTable) -> str | None:
    """Return the column an ``EventsTable`` should be split on, or ``None`` for a single event.

    A core (pynwb 4.0) ``EventsTable`` always has a ``timestamp`` column and may have a
    ``duration`` column; any remaining column is a value column carrying a per-event
    category (e.g. ``annotation`` from a hand-built file, or ``strobe`` / ``event_type``
    written by NeuroConv's events interfaces). We split the table into one event per
    unique value of that column.

    Returns
    -------
    str or None
        The name of the single value column to split on, or ``None`` when the table has
        no value column (it then yields a single event named after the table).

    Raises
    ------
    NotImplementedError
        If the table has more than one value column. Splitting on a combination of
        columns is not supported.
    """
    value_columns = [column for column in table.colnames if column not in ("timestamp", "duration")]
    if len(value_columns) == 0:
        return None
    elif len(value_columns) == 1:
        return value_columns[0]
    else:
        raise NotImplementedError(
            f"EventsTable {table.name!r} has multiple value columns {value_columns!r}, but splitting "
            "events on more than one column is not supported. If you need multi-column splitting, "
            "please raise an issue at https://github.com/LernerLab/GuPPy/issues."
        )


def _discover_core_events_tables(nwbfile: NWBFile, seen_names: set[str]) -> list[str]:
    """Discover event names from core (pynwb 4.0) ``EventsTable`` objects.

    Each ``EventsTable`` in ``nwbfile.events`` maps to one or more events. A table with a
    single value column (any column other than ``timestamp``/``duration``) yields one event
    per unique value of that column (``{table_name}_{value}``); a table without one yields a
    single event ``{table_name}``.
    """
    events = []
    for table_name, table in (getattr(nwbfile, "events", None) or {}).items():
        _register_unique_name(seen_names, table_name, "EventsTable")
        split_column = _core_events_split_column(table)
        if split_column is None:
            events.append(table_name)
        else:
            for value in np.unique(np.asarray(table[split_column][:])):
                events.append(f"{table_name}_{value}")
    return events


def _build_event_index_v02(nwbfile: NWBFile) -> dict[str, tuple]:
    """Build a unified event index for ndx-events v0.2 objects.

    Returns
    -------
    dict
        Mapping of event name to tagged source tuple:
        ``("annotated", object, row_index)``,
        ``("labeled", object, label_index)``, or
        ``("events", object)``.
    """
    index = {}
    for neurodata_object in nwbfile.objects.values():
        neurodata_type = getattr(neurodata_object, "neurodata_type", None)
        if neurodata_type == "AnnotatedEventsTable":
            for row_index, label in enumerate(neurodata_object["label"].data):
                index[f"{neurodata_object.name}_{label}"] = ("annotated", neurodata_object, row_index)
        elif neurodata_type == "LabeledEvents":
            for label_index, label in enumerate(neurodata_object.data__labels):
                index[f"{neurodata_object.name}_{label}"] = ("labeled", neurodata_object, label_index)
        elif neurodata_type == "Events":
            index[neurodata_object.name] = ("events", neurodata_object)
    return index


def _build_core_events_index(nwbfile: NWBFile) -> dict[str, tuple]:
    """Build a unified event index for core (pynwb 4.0) ``EventsTable`` objects.

    Returns
    -------
    dict
        Mapping of event name to ``("core", table, split_column_or_None, value_or_None)``.
    """
    index = {}
    for table_name, table in (getattr(nwbfile, "events", None) or {}).items():
        split_column = _core_events_split_column(table)
        if split_column is None:
            index[table_name] = ("core", table, None, None)
        else:
            for value in np.unique(np.asarray(table[split_column][:])):
                index[f"{table_name}_{value}"] = ("core", table, split_column, value)
    return index


def _read_ndx_event(*, event_name: str, source_info: tuple) -> dict[str, object]:
    """Read timestamps for a single event from its indexed source.

    Parameters
    ----------
    event_name : str
        The event name (used as ``storename``).
    source_info : tuple
        Tagged tuple from ``_build_event_index_v02`` (ndx-events v0.2) or
        ``_build_core_events_index`` (core ``EventsTable``).

    Returns
    -------
    dict
        ``{"storename": ..., "timestamps": ...}``
    """
    tag = source_info[0]

    if tag == "annotated":
        _, neurodata_object, row_index = source_info
        return {"storename": event_name, "timestamps": np.array(neurodata_object["event_times"][row_index])}

    if tag == "labeled":
        _, neurodata_object, label_index = source_info
        all_timestamps = np.array(neurodata_object.timestamps[:])
        all_data = np.array(neurodata_object.data[:])
        return {"storename": event_name, "timestamps": all_timestamps[all_data == label_index]}

    if tag == "events":
        _, neurodata_object = source_info
        return {"storename": event_name, "timestamps": np.array(neurodata_object.timestamps[:])}

    if tag == "core":
        _, table, split_column, split_value = source_info
        all_timestamps = np.array(table["timestamp"][:])
        if split_column is not None:
            all_values = np.asarray(table[split_column][:])
            return {"storename": event_name, "timestamps": all_timestamps[all_values == split_value]}
        return {"storename": event_name, "timestamps": all_timestamps}

    raise ValueError(
        f"Unknown event source tag: {tag!r}. Expected one of 'annotated', 'labeled', 'events', 'core'. "
        "This indicates an internal inconsistency in the NWB event index."
    )


def _find_nwb_file(folder_path: str | Path) -> str:
    """Return the single NWB file path in *folder_path*, raising if not exactly one."""
    nwb_paths = list(Path(folder_path).glob("*.nwb"))
    if len(nwb_paths) > 1:
        raise ValueError(
            f"Multiple NWB files found in '{folder_path}': {sorted(str(p) for p in nwb_paths)}. "
            "Each session folder must contain exactly one .nwb file."
        )
    if len(nwb_paths) == 0:
        raise FileNotFoundError(f"No NWB file found in '{folder_path}'.")
    return str(nwb_paths[0])


def _get_ndx_events_version(io: NWBHDF5IO) -> str | None:
    """Return the ndx-events namespace version string from an open NWBHDF5IO, or None.

    Parameters
    ----------
    io : pynwb.NWBHDF5IO
        An open (already-read) NWBHDF5IO instance.

    Returns
    -------
    str or None
        Version string such as ``"0.2.2"`` or ``"0.4.0"``, or ``None`` if the
        ndx-events namespace is not present in the file.
    """
    namespace_catalog = io.manager.namespace_catalog
    if "ndx-events" not in namespace_catalog.namespaces:
        return None
    return namespace_catalog.get_namespace("ndx-events").version


def _parse_event_name(event: str, series_name_to_object: dict[str, object]) -> tuple[str, int | None]:
    """
    Resolve an event name to a series name and optional column index.

    Parameters
    ----------
    event : str
        Event name as produced by ``discover_events_and_flags``.
    series_name_to_object : dict
        Mapping of series name to ``FiberPhotometryResponseSeries`` object.

    Returns
    -------
    series_name : str
    column_index : int or None
        ``None`` for single-channel (1-D) series.
    """
    if event in series_name_to_object:
        return event, None

    # Parse "{series_name}_{column_index}" format
    parts = event.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        series_name, column_index = parts[0], int(parts[1])
        if series_name in series_name_to_object:
            return series_name, column_index

    available_series = sorted(series_name_to_object.keys())
    raise ValueError(
        f"Event '{event}' could not be resolved to any FiberPhotometryResponseSeries in the NWB file. "
        f"Available series: {available_series}. Multi-channel series accept '{{series_name}}_{{column_index}}' "
        "form (e.g. 'fp_signal_0')."
    )


def _resolve_timing(series: object, num_samples: int) -> tuple[float, np.ndarray]:
    """
    Derive sampling rate and a full timestamps array from an NWB TimeSeries.

    Checks ``rate``, then ``timestamps``. If ``rate`` is present but no explicit
    timestamps, reconstructs timestamps from ``starting_time`` and ``rate``.

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
    if series.timestamps is not None:
        timestamps = np.array(series.timestamps[:])
        sampling_rate = 1.0 / float(np.median(np.diff(timestamps)))
    else:
        starting_time = series.starting_time or 0.0
        timestamps = starting_time + np.arange(num_samples) / series.rate
        sampling_rate = float(series.rate)

    return sampling_rate, timestamps
