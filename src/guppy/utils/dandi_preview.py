"""Open one DANDI NWB asset and report what it holds.

Where :mod:`guppy.utils.dandi_filter` answers yes or no, this reads the detail the source panel
shows before a file is chosen: the channels of each fiber photometry response series, the brain
region, indicator and wavelengths behind each one, the store names Step 1 will ask for labels
against, the events a PSTH could be aligned to, and a decimated slice of every trace.

Everything is read from the file's HDF5 structure over the network, so the entry points take a
content URL rather than a path and never download the asset.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from math import ceil

import h5py
import numpy as np
from dandi.dandiapi import DandiAPIClient

from .dandi_filter import (
    find_fiber_photometry_container,
    find_photometry_series,
    find_typed_child,
    series_containers,
)

logger = logging.getLogger(__name__)

# How many samples a preview trace is decimated to before it is plotted.
DEFAULT_TRACE_POINTS = 2000
# Seconds of each channel a preview reads off the front of the recording.
DEFAULT_TRACE_DURATION_IN_SECONDS = 60.0

# Excitation wavelengths at or below this are isosbestic for the green calcium and dopamine
# sensors, whose isosbestic point sits near 405-415 nm. Used only to suggest a GuPPy store label.
ISOSBESTIC_WAVELENGTH_CEILING_IN_NM = 420.0

FIBER_PHOTOMETRY_TABLE = "FiberPhotometryTable"
# Event containers GuPPy's NWB reader turns into PSTH event stores: the ndx-events v0.2 types
# and the core ``EventsTable`` that replaced them.
EVENT_NEURODATA_TYPES = frozenset({"Events", "EventsTable", "LabeledEvents", "AnnotatedEventsTable"})


@dataclass(frozen=True)
class ChannelInfo:
    """One fiber photometry channel: a column of one response series, and the fiber behind it.

    Attributes
    ----------
    store_name : str
        The store id GuPPy's Label Stores page will show for this channel.
    series_name : str
        Name of the ``FiberPhotometryResponseSeries`` the channel belongs to.
    column_index : int or None
        Column within a multi-channel series; None for a single-channel series.
    location : str or None
        Recording site, from the fiber photometry table's ``location`` column.
    indicator : str or None
        Indicator label, from the ``Indicator`` object the table row points at.
    excitation_wavelength_in_nm, emission_wavelength_in_nm : float or None
        Wavelengths for the channel, where the file records them.
    suggested_label : str or None
        A ``signal_<site>`` / ``control_<site>`` store label consistent with the channel's
        excitation wavelength, or None when the wavelength or the site is unknown.
    """

    store_name: str
    series_name: str
    column_index: int | None
    location: str | None
    indicator: str | None
    excitation_wavelength_in_nm: float | None
    emission_wavelength_in_nm: float | None
    suggested_label: str | None


@dataclass(frozen=True)
class SeriesInfo:
    """Timing and shape of one ``FiberPhotometryResponseSeries``.

    Attributes
    ----------
    name : str
        Object name within the NWB file.
    sample_count : int
        Length of the series' leading dimension.
    channel_count : int
        Number of columns; 1 for a single-channel series.
    sampling_rate_in_hz : float or None
        Rate, when the series stores a regular ``starting_time``/``rate`` pair.
    duration_in_seconds : float or None
        Recording length, from the rate or from the timestamps' span.
    """

    name: str
    sample_count: int
    channel_count: int
    sampling_rate_in_hz: float | None
    duration_in_seconds: float | None


@dataclass(frozen=True)
class PhotometryProbe:
    """What one NWB file holds, read from its header.

    Attributes
    ----------
    series : tuple of SeriesInfo
        Every ``FiberPhotometryResponseSeries`` in the file.
    channels : tuple of ChannelInfo
        One entry per channel of every series, in GuPPy store order.
    event_names : tuple of str
        Names of the event containers GuPPy can align a PSTH to.
    session_description, session_start_time, identifier : str or None
        The file's own session-level fields.
    subject : dict of {str: str}
        The ``Subject`` group's fields, as strings.
    """

    series: tuple[SeriesInfo, ...] = ()
    channels: tuple[ChannelInfo, ...] = ()
    event_names: tuple[str, ...] = ()
    session_description: str | None = None
    session_start_time: str | None = None
    identifier: str | None = None
    subject: dict[str, str] = field(default_factory=dict)

    @property
    def has_photometry(self) -> bool:
        """Whether the file holds any fiber photometry response series."""
        return bool(self.series)

    @property
    def locations(self) -> tuple[str, ...]:
        """Distinct recording sites across the file's channels, in channel order."""
        return _unique(channel.location for channel in self.channels)

    @property
    def indicators(self) -> tuple[str, ...]:
        """Distinct indicators across the file's channels, in channel order."""
        return _unique(channel.indicator for channel in self.channels)


def _unique(values: Iterable[str | None]) -> tuple[str, ...]:
    """Return the non-empty values in order, without duplicates."""
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return tuple(seen)


def _decode(value: object) -> str | None:
    """Render an HDF5 scalar as a plain string, decoding bytes."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _read_scalar_dataset(file: h5py.File, path: str) -> str | None:
    """Read a scalar dataset as a string, or None when the file does not have it."""
    dataset = file.get(path)
    return None if dataset is None else _decode(dataset[()])


def _find_fiber_photometry_table(file: h5py.File) -> h5py.Group | None:
    """Return the file's ``FiberPhotometryTable``, whatever its author named it or its parent."""
    return find_typed_child(find_fiber_photometry_container(file), FIBER_PHOTOMETRY_TABLE)


def _find_event_names(file: h5py.File) -> tuple[str, ...]:
    """Return the names of the event containers GuPPy can align a PSTH to.

    Looks wherever the supported event types land: ndx-events objects sit in ``/acquisition``
    or a processing module, while a core ``EventsTable`` sits in the file's ``/events`` group.
    """
    containers = series_containers(file)
    events_group = file.get("events")
    if events_group is not None:
        containers.append(events_group)
    names = []
    for group in containers:
        for name, member in group.items():
            if member.attrs.get("neurodata_type") in EVENT_NEURODATA_TYPES and name not in names:
                names.append(name)
    return tuple(names)


def _read_fiber_photometry_table(file: h5py.File) -> list[dict[str, object]]:
    """Read the fiber photometry table into one dictionary per row.

    Each row describes one fiber: where it sat, which indicator it recorded, and the
    wavelengths involved. The indicator and excitation source are object references, and
    their labels and wavelengths live as attributes on the objects they point at.
    """
    table = _find_fiber_photometry_table(file)
    if table is None:
        return []
    locations = table["location"][:] if "location" in table else []
    rows = []
    for row_index in range(len(locations)):
        indicator = _referenced_attribute(file=file, table=table, column="indicator", row=row_index, key="label")
        excitation = _column_value(table=table, column="excitation_wavelength_in_nm", row=row_index)
        if excitation is None:
            excitation = _referenced_attribute(
                file=file,
                table=table,
                column="excitation_source",
                row=row_index,
                key="excitation_wavelength_in_nm",
            )
        rows.append(
            {
                "location": _decode(locations[row_index]),
                "indicator": _decode(indicator),
                "excitation_wavelength_in_nm": _as_float(excitation),
                "emission_wavelength_in_nm": _as_float(
                    _column_value(table=table, column="emission_wavelength_in_nm", row=row_index)
                ),
            }
        )
    return rows


def _column_value(*, table: h5py.Group, column: str, row: int) -> object:
    """Read one cell of a dynamic-table column, or None when the column is absent."""
    dataset = table.get(column)
    return None if dataset is None else dataset[row]


def _referenced_attribute(*, file: h5py.File, table: h5py.Group, column: str, row: int, key: str) -> object:
    """Follow a table cell's object reference and read ``key`` off the object it points at."""
    reference = _column_value(table=table, column=column, row=row)
    if reference is None:
        return None
    return file[reference].attrs.get(key)


def _as_float(value: object) -> float | None:
    """Render an HDF5 numeric scalar as a float, or None when it is missing."""
    return None if value is None else float(value)


def _suggested_label(*, location: str | None, excitation_wavelength_in_nm: float | None) -> str | None:
    """Name the GuPPy store label a channel's wavelength and site imply."""
    if not location or excitation_wavelength_in_nm is None:
        return None
    role = "control" if excitation_wavelength_in_nm <= ISOSBESTIC_WAVELENGTH_CEILING_IN_NM else "signal"
    return f"{role}_{location}"


def _series_timing(series: h5py.Group, sample_count: int) -> tuple[float | None, float | None]:
    """Return the series' sampling rate and duration, from whichever timing it stores."""
    starting_time = series.get("starting_time")
    if starting_time is not None:
        rate = starting_time.attrs.get("rate")
        if rate:
            return float(rate), sample_count / float(rate)
    timestamps = series.get("timestamps")
    if timestamps is not None and len(timestamps) > 1:
        first, last = float(timestamps[0]), float(timestamps[-1])
        span = last - first
        return (len(timestamps) - 1) / span if span > 0 else None, span
    return None, None


def probe_photometry(*, file: h5py.File) -> PhotometryProbe:
    """Report the fiber photometry content of an open NWB file.

    Reads only header metadata -- object names, dataset shapes, table columns and device
    attributes -- so the call is affordable against a file streamed from the archive.

    Parameters
    ----------
    file : h5py.File
        An open NWB file, local or backed by a remote reader.

    Returns
    -------
    PhotometryProbe
        The file's response series, per-channel fiber details, event containers and
        session-level fields. ``has_photometry`` is False for a file that holds none.
    """
    table_rows = _read_fiber_photometry_table(file)
    series_infos = []
    channels = []
    for name, series in find_photometry_series(file).items():
        data = series["data"]
        sample_count = int(data.shape[0])
        channel_count = int(data.shape[1]) if data.ndim == 2 else 1
        sampling_rate, duration = _series_timing(series, sample_count)
        series_infos.append(
            SeriesInfo(
                name=name,
                sample_count=sample_count,
                channel_count=channel_count,
                sampling_rate_in_hz=sampling_rate,
                duration_in_seconds=duration,
            )
        )
        region = series.get("fiber_photometry_table_region")
        row_indices = [int(index) for index in region[:]] if region is not None else []
        for column_index in range(channel_count):
            # A single-channel series keeps its own name as the store id; a multi-channel one
            # contributes one store per column, which is how GuPPy's NWB reader names them.
            store_name = name if data.ndim == 1 else f"{name}_{column_index}"
            row = table_rows[row_indices[column_index]] if column_index < len(row_indices) else {}
            channels.append(
                ChannelInfo(
                    store_name=store_name,
                    series_name=name,
                    column_index=None if data.ndim == 1 else column_index,
                    location=row.get("location"),
                    indicator=row.get("indicator"),
                    excitation_wavelength_in_nm=row.get("excitation_wavelength_in_nm"),
                    emission_wavelength_in_nm=row.get("emission_wavelength_in_nm"),
                    suggested_label=_suggested_label(
                        location=row.get("location"),
                        excitation_wavelength_in_nm=row.get("excitation_wavelength_in_nm"),
                    ),
                )
            )

    subject = file.get("general/subject")
    return PhotometryProbe(
        series=tuple(series_infos),
        channels=tuple(channels),
        event_names=_find_event_names(file),
        session_description=_read_scalar_dataset(file, "session_description"),
        session_start_time=_read_scalar_dataset(file, "session_start_time"),
        identifier=_read_scalar_dataset(file, "identifier"),
        subject={} if subject is None else {key: _decode(subject[key][()]) for key in subject},
    )


@dataclass(frozen=True)
class ExampleTraces:
    """A decimated slice off the front of one response series, ready to plot.

    Attributes
    ----------
    series_name : str
        The series the slice came from.
    timestamps : numpy.ndarray
        Times in seconds, one per returned sample.
    traces : dict of {str: numpy.ndarray}
        Store name to that channel's samples.
    """

    series_name: str
    timestamps: np.ndarray
    traces: dict[str, np.ndarray]


def read_example_traces(
    *,
    file: h5py.File,
    probe: PhotometryProbe,
    series_name: str | None = None,
    duration_in_seconds: float = DEFAULT_TRACE_DURATION_IN_SECONDS,
    max_points: int = DEFAULT_TRACE_POINTS,
) -> ExampleTraces:
    """Read the opening seconds of one response series, decimated for display.

    Parameters
    ----------
    file : h5py.File
        The open NWB file ``probe`` was taken from.
    probe : PhotometryProbe
        The file's probe, which supplies the series' timing and channel names.
    series_name : str or None, optional
        Series to read. Defaults to the file's first response series.
    duration_in_seconds : float, optional
        How much of the recording to read from its start. A series whose file records no
        sampling rate is read whole, since there is then nothing to convert seconds into
        samples with.
    max_points : int, optional
        Upper bound on the samples returned per channel; the slice is strided down to it.

    Returns
    -------
    ExampleTraces
        Timestamps and one trace per channel of the chosen series.
    """
    series_info = next(info for info in probe.series if series_name in (None, info.name))
    series = find_photometry_series(file)[series_info.name]
    data = series["data"]

    sample_count = series_info.sample_count
    if series_info.sampling_rate_in_hz:
        sample_count = min(sample_count, int(duration_in_seconds * series_info.sampling_rate_in_hz))
    sample_count = max(sample_count, 1)
    step = max(1, ceil(sample_count / max_points))

    values = data[:sample_count:step]
    timestamps_dataset = series.get("timestamps")
    if timestamps_dataset is not None:
        timestamps = np.asarray(timestamps_dataset[:sample_count:step], dtype=float)
    else:
        starting_time = series.get("starting_time")
        offset = float(starting_time[()]) if starting_time is not None else 0.0
        rate = series_info.sampling_rate_in_hz or 1.0
        timestamps = offset + np.arange(len(values)) * step / rate

    channels = [channel for channel in probe.channels if channel.series_name == series_info.name]
    traces = {
        channel.store_name: np.asarray(values if values.ndim == 1 else values[:, channel.column_index], dtype=float)
        for channel in channels
    }
    return ExampleTraces(series_name=series_info.name, timestamps=timestamps, traces=traces)


@dataclass(frozen=True)
class AssetPreview:
    """A probed asset and the example traces read in the same streaming session.

    Attributes
    ----------
    dandiset_id, asset_path : str
        Where the asset came from.
    probe : PhotometryProbe
        What the file holds.
    traces : ExampleTraces or None
        Opening slice of the file's first response series; None when it holds no photometry.
    """

    dandiset_id: str
    asset_path: str
    probe: PhotometryProbe
    traces: ExampleTraces | None


# Excluded from coverage: resolves the URL over the network. Exercised by the local-only live
# suite tests/unit/utils/test_dandi_preview_live.py (marker ``dandi_live``, deselected in CI).
def asset_content_url(*, dandiset_id: str, asset_path: str, version: str = "draft") -> str:  # pragma: no cover
    """Resolve the S3 URL an asset's bytes can be range-read from.

    Reading a public asset this way needs no DANDI API key: the redirect target is a public
    S3 object.

    Parameters
    ----------
    dandiset_id : str
        Six-digit dandiset ID.
    asset_path : str
        Path of the asset within the dandiset.
    version : str, optional
        Dandiset version the asset belongs to.

    Returns
    -------
    str
        A directly readable URL for the asset's bytes.
    """
    with DandiAPIClient() as client:
        asset = client.get_dandiset(dandiset_id, version).get_asset_by_path(asset_path)
        return asset.get_content_url(follow_redirects=1, strip_query=False)


# Excluded from coverage: opens a network connection to the archive. Exercised by the local-only
# live suite (marker ``dandi_live``, deselected in CI); offline tests call the probe and trace
# readers against local NWB files instead.
def open_remote_hdf5(*, content_url: str) -> h5py.File:  # pragma: no cover
    """Open an HDF5 file over HTTP range requests, without downloading it.

    Parameters
    ----------
    content_url : str
        Directly readable URL for the file's bytes, as
        :func:`asset_content_url` returns.

    Returns
    -------
    h5py.File
        An open file the caller owns and must close.
    """
    import remfile

    return h5py.File(remfile.File(content_url), mode="r")


# Excluded from coverage: streams from the archive. Exercised by the local-only live suite
# tests/unit/utils/test_dandi_preview_live.py (marker ``dandi_live``, deselected in CI); the
# probe and trace readers it composes are unit-tested against local NWB files.
def preview_asset(  # pragma: no cover
    *,
    dandiset_id: str,
    asset_path: str,
    version: str = "draft",
    series_name: str | None = None,
    trace_duration_in_seconds: float = DEFAULT_TRACE_DURATION_IN_SECONDS,
    max_points: int = DEFAULT_TRACE_POINTS,
) -> AssetPreview:
    """Stream one asset's header and the opening seconds of its traces.

    Both reads share a single streaming session, so the preview costs one file open rather
    than two. No DANDI API key is needed for a public dandiset.

    Parameters
    ----------
    dandiset_id : str
        Six-digit dandiset ID.
    asset_path : str
        Path of the asset within the dandiset.
    version : str, optional
        Dandiset version the asset belongs to.
    series_name : str or None, optional
        Response series to read the traces from. Defaults to the file's first.
    trace_duration_in_seconds : float, optional
        Seconds of the recording to read for the example traces.
    max_points : int, optional
        Upper bound on the samples returned per channel.

    Returns
    -------
    AssetPreview
        The probe, and the example traces when the asset holds photometry.
    """
    content_url = asset_content_url(dandiset_id=dandiset_id, asset_path=asset_path, version=version)
    file = open_remote_hdf5(content_url=content_url)
    try:
        probe = probe_photometry(file=file)
        traces = (
            read_example_traces(
                file=file,
                probe=probe,
                series_name=series_name,
                duration_in_seconds=trace_duration_in_seconds,
                max_points=max_points,
            )
            if probe.has_photometry
            else None
        )
    finally:
        file.close()
    return AssetPreview(dandiset_id=dandiset_id, asset_path=asset_path, probe=probe, traces=traces)
