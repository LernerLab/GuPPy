import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from guppy.extractors import BaseRecordingExtractor
from guppy.extractors.detect_acquisition_formats import _classify_csv_file
from guppy.utils._hdf5_io import write_hdf5

# Filename patterns of per-channel and per-event files derived from a session, which are not
# raw NPM sources.
DERIVED_FILENAME_PATTERNS = ("*chev*", "*chod*", "*chpr*", "event*")
# Substring, lowercased, that marks a column as a clock.
TIMESTAMP_COLUMN_SUBSTRING = "timestamp"
# Names of the column recording which excitation LEDs lit each frame, lowercased.
STATE_COLUMN_NAMES = frozenset({"flags", "ledstate"})
# The three lowest bits of a state word are one flag per excitation LED; the higher bits are
# digital TTL lines. A wavelength's frames are those whose word has that wavelength's bit set.
EXCITATION_BITS = 0b111
WAVELENGTH_TO_EXCITATION_CODE = {415: 1, 470: 2, 560: 4}
# Lowercased names of every column an NPM file writes that is not a region: the clocks and frame
# index, the state word, and the digital lines.
NON_REGION_COLUMN_NAMES = frozenset(
    {
        "framecounter",
        "timestamp",
        "systemtimestamp",
        "computertimestamp",
        "triggerevents",
        "flags",
        "ledstate",
        "stimulation",
        "output0",
        "output1",
        "input0",
        "input1",
    }
)
# pandas names a column whose header cell is blank ``Unnamed: {position}``.
BLANK_HEADER_COLUMN_PATTERN = re.compile(r"Unnamed: \d+")
# Names of the interleaved channel slots of a file with no state column, in cycle order.
STRIDE_CHANNEL_SLOTS = ("chev", "chod", "chpr")
# Timestamp units a session can be recorded in, and the factor converting each to seconds.
TIME_UNIT_DIVISORS = {"seconds": 1.0, "milliseconds": 1e3, "microseconds": 1e6}
DEFAULT_TIME_UNIT = "seconds"
# Channel count assumed for a file with no state column until the Label Stores page asks for one.
DEFAULT_NUM_CHANNELS = 2


def _is_number(value: object) -> bool:
    """
    Check whether a value parses as a float.

    Parameters
    ----------
    value : object
        Value to test, typically a CSV cell or header label.

    Returns
    -------
    bool
        ``True`` if ``float(value)`` succeeds, ``False`` otherwise.
    """
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


@dataclass(frozen=True)
class EventStore:
    """
    Where an event store comes from.

    Attributes
    ----------
    file : str
        Name of the event file.
    event_value : object
        The value whose events the store holds, or ``None`` when the file is not split and
        the store holds all of them.
    """

    file: str
    event_value: object


@dataclass(frozen=True)
class ChannelStore:
    """
    Where a photometry channel comes from.

    Attributes
    ----------
    file : str
        Name of the data file.
    excitation_wavelength_in_nm : int or None
        Wavelength of the LED that lit the channel's frames, or ``None`` for a file with no
        state column.
    interleave_position : int or None
        The channel's slot in the interleave cycle of a file with no state column, or
        ``None`` for a file with one.
    data_column : str or int
        Column the channel's samples are read from; a position in a header-less file.
    timestamp_column : str or int
        Column the channel is timed by; ``0`` in a header-less file.
    """

    file: str
    excitation_wavelength_in_nm: int | None
    interleave_position: int | None
    data_column: str | int
    timestamp_column: str | int


class NpmRecordingExtractor(BaseRecordingExtractor):
    """
    Extractor for fiber photometry data from Neurophotometrics (NPM) systems.

    Parameters
    ----------
    folder_path : str
        Path to the session folder containing the NPM CSV files.
    num_ch : int, optional
        Number of interleaved channels expected. Default is 2.
    npm_timestamp_column_name : str, optional
        Timestamp column to use in files that have more than one.
    npm_time_unit : str, optional
        Unit of the session's timestamps (``"seconds"``, ``"milliseconds"``, or
        ``"microseconds"``).
    npm_split_events : dict of str to bool, optional
        Maps each event file's name to whether its events are split into one event stream
        per unique value.
    """

    def __init__(
        self,
        folder_path: str,
        *,
        num_ch: int = DEFAULT_NUM_CHANNELS,
        npm_timestamp_column_name: str | None = None,
        npm_time_unit: str | None = None,
        npm_split_events: dict[str, bool] | None = None,
    ) -> None:
        self.folder_path = folder_path
        self.num_ch = num_ch
        self.npm_timestamp_column_name = npm_timestamp_column_name
        self.npm_time_unit = npm_time_unit
        self.npm_split_events = npm_split_events
        self._dataframes: dict[str, pd.DataFrame] | None = None
        self._records: dict[str, EventStore | ChannelStore] | None = None

    @classmethod
    def discover_events_and_flags(
        cls, folder_path: str, num_ch: int, inputParameters: dict[str, object] | None
    ) -> tuple[list[str], list[str]]:
        """
        Discover the session's store IDs and format flags.

        Parameters
        ----------
        folder_path : str
            Path to the session folder.
        num_ch : int
            Number of interleaved channels in a file with no state column.
        inputParameters : dict, optional
            Input parameters carrying the NPM configuration: ``npm_timestamp_column_name``, and
            ``npm_split_events`` mapping each event file's name to whether it is split by value.

        Returns
        -------
        events : list of str
            Store IDs of every stream the session yields, in file order.
        flags : list of str
            One format flag per raw source file: ``"event_np"``, ``"data_np_v2"`` (a
            state column names each frame's LED) or ``"data_np"`` (it does not).
        """
        if inputParameters is None:
            inputParameters = {}
        dataframes = cls._read_source_files(folder_path)
        store_records = cls._store_records(
            dataframes,
            num_ch=num_ch,
            npm_timestamp_column_name=inputParameters.get("npm_timestamp_column_name"),
            npm_split_events=inputParameters.get("npm_split_events"),
        )
        store_ids = list(store_records)

        flags = []
        for df in dataframes.values():
            flags.append(cls._layout_flag(df))
        return store_ids, flags

    @classmethod
    def has_multiple_event_ttls(cls, folder_path: str) -> dict[str, bool]:
        """
        Check which of the session's event files hold more than one TTL value.

        Parameters
        ----------
        folder_path : str
            Path to the session folder.

        Returns
        -------
        dict of str to bool
            Maps each event file's name to whether its value column holds more than one
            distinct value.
        """
        multiple_event_ttls = {}
        for file_name, df in cls._read_source_files(folder_path).items():
            if not cls._is_event_file(df):
                continue
            event_values = df.iloc[:, 1]
            unique_event_values = np.unique(event_values)
            multiple_event_ttls[file_name] = len(unique_event_values) > 1
        return multiple_event_ttls

    @classmethod
    def get_timestamp_column_options(cls, folder_path: str) -> list[str]:
        """
        List the timestamp columns the session's data files offer.

        Parameters
        ----------
        folder_path : str
            Path to the session folder.

        Returns
        -------
        list of str
            Distinct timestamp column names, in the order encountered. Empty for a session
            whose data files are all header-less.
        """
        column_options = []
        for df in cls._read_source_files(folder_path).values():
            if cls._is_event_file(df):
                continue
            if not cls._has_text_header(df):
                continue
            for name in cls._timestamp_column_names(df):
                if name not in column_options:
                    column_options.append(name)
        return column_options

    def get_store_provenance(self) -> dict[str, dict[str, object]]:
        """
        Report what each data channel was demultiplexed from.

        Returns
        -------
        dict
            Maps each photometry channel's store ID to the fields of its :class:`ChannelStore`.
        """
        store_records = self._get_records()
        provenance = {}
        for store_id, record in store_records.items():
            if isinstance(record, ChannelStore):
                provenance[store_id] = asdict(record)
        return provenance

    def read(self, *, events: list[str], outputPath: str) -> list[dict[str, Any]]:
        """
        Read the requested stores.

        Parameters
        ----------
        events : list of str
            Store IDs to read.
        outputPath : str
            Path to the output directory. Unused; required by the base-class interface.

        Returns
        -------
        list of dict
            One dict per store, in the order requested. An event store carries ``store_id``
            and ``timestamps``; a photometry channel adds ``data`` and ``sampling_rate``.
            Timestamps are in seconds, and each channel is timed by its own frames.

        Raises
        ------
        ValueError
            If ``npm_time_unit`` is not one of the recognized units.
        """
        store_records = self._get_records()
        dataframes = self._get_dataframes()
        time_unit = self.npm_time_unit or DEFAULT_TIME_UNIT
        if time_unit not in TIME_UNIT_DIVISORS:
            raise ValueError(
                f"npm_time_unit='{time_unit}' is not a recognized timestamp unit; choose one of "
                f"{list(TIME_UNIT_DIVISORS)}."
            )
        divisor = TIME_UNIT_DIVISORS[time_unit]

        output_dicts = []
        for store_id in events:
            record = store_records[store_id]
            df = dataframes[record.file]

            if isinstance(record, EventStore):
                rows = self._event_rows(df, record)
                timestamps = np.asarray(df.iloc[:, 0], dtype=float)[rows] / divisor
                output_dicts.append({"store_id": store_id, "timestamps": timestamps})
                continue

            rows = self._channel_rows(df, record)
            timestamps = np.asarray(df[record.timestamp_column], dtype=float)[rows] / divisor
            data = np.asarray(df[record.data_column], dtype=float)[rows]
            sampling_rate = timestamps.shape[0] / (timestamps[-1] - timestamps[0])
            output_dicts.append(
                {
                    "store_id": store_id,
                    "timestamps": timestamps,
                    "data": data,
                    "sampling_rate": np.array([sampling_rate]),
                }
            )
        return output_dicts

    def save(self, *, output_dicts: list[dict[str, Any]], outputPath: str) -> None:
        """
        Write stores to HDF5.

        Parameters
        ----------
        output_dicts : list of dict
            Output dicts as returned by :meth:`read`.
        outputPath : str
            Path to the output directory. Each store is written to ``{store_id}.hdf5``, one
            dataset per array.
        """
        for output_dict in output_dicts:
            store_id = output_dict["store_id"]
            for key, value in output_dict.items():
                if key == "store_id":
                    continue
                write_hdf5(value, store_id, outputPath, key)

    def count_samples(self, *, event: str) -> int:
        """
        Count the samples in one store.

        Parameters
        ----------
        event : str
            Store ID to count.

        Returns
        -------
        int
            Number of samples in the store.
        """
        record = self._get_records()[event]
        df = self._get_dataframes()[record.file]
        if isinstance(record, EventStore):
            rows = self._event_rows(df, record)
        else:
            rows = self._channel_rows(df, record)
        return len(rows)

    def stub(self, *, folder_path: str | Path, duration_in_seconds: float = 1.0) -> None:
        """
        Write a copy of the session folder with each raw NPM file truncated.

        Each data file is cut at its own first timestamp plus ``duration_in_seconds``, read on
        the file's first timestamp column (its leading column when it has no header). A first
        timestamp above ``1e6`` is taken to be in milliseconds. Each event file then keeps only
        the events that fall inside the span a truncated data file retains on any of its
        timestamp columns, so no event outlasts the photometry it belongs to; an event file left
        with no events is removed.

        Parameters
        ----------
        folder_path : str or Path
            Destination directory. Overwritten if it exists.
        duration_in_seconds : float, optional
            Approximate duration of data to retain. Default is 1.0.
        """
        folder_path = Path(folder_path)
        if folder_path.exists():
            shutil.rmtree(folder_path)
        shutil.copytree(self.folder_path, folder_path)

        retained_spans = []
        event_csv_paths = []
        for csv_path in sorted(folder_path.glob("*.csv")):
            if _classify_csv_file(str(csv_path)) != "npm":
                continue
            df = pd.read_csv(csv_path, index_col=False)
            has_text_header = self._has_text_header(df)
            if not has_text_header:
                df = pd.read_csv(csv_path, header=None)
            if self._is_event_file(df):
                event_csv_paths.append(csv_path)
                continue

            if has_text_header:
                timestamp_column_names = self._timestamp_column_names(df)
                if not timestamp_column_names:
                    timestamp_column_names = [df.columns[0]]
            else:
                timestamp_column_names = [0]
            timestamp_column = timestamp_column_names[0]

            first_timestamp = float(df[timestamp_column].iloc[0])
            if first_timestamp > 1e6:
                unit_factor = 1000.0
            else:
                unit_factor = 1.0
            cutoff = first_timestamp + duration_in_seconds * unit_factor
            df = df[df[timestamp_column] <= cutoff]
            df.to_csv(csv_path, index=False, header=has_text_header)

            for name in timestamp_column_names:
                retained_spans.append((float(df[name].min()), float(df[name].max())))

        for csv_path in event_csv_paths:
            df = pd.read_csv(csv_path, header=None)
            event_timestamps = np.asarray(df.iloc[:, 0], dtype=float)
            inside_retained_data = np.zeros(event_timestamps.shape[0], dtype=bool)
            for span_start, span_end in retained_spans:
                inside_retained_data |= (event_timestamps >= span_start) & (event_timestamps <= span_end)
            if inside_retained_data.any():
                df[inside_retained_data].to_csv(csv_path, index=False, header=False)
            else:
                csv_path.unlink()

    def _get_dataframes(self) -> dict[str, pd.DataFrame]:
        """
        Get the session's raw files, read once per instance.

        Returns
        -------
        dict of str to pd.DataFrame
            Maps each raw file's name to its contents, as read by :meth:`_read_source_files`.
        """
        if self._dataframes is None:
            self._dataframes = self._read_source_files(self.folder_path)
        return self._dataframes

    def _get_records(self) -> dict[str, EventStore | ChannelStore]:
        """
        Get the session's store records under this instance's configuration, built once.

        Returns
        -------
        dict of str to EventStore or ChannelStore
            Maps each store ID to its record, as built by :meth:`_store_records`.
        """
        if self._records is None:
            self._records = self._store_records(
                self._get_dataframes(),
                num_ch=self.num_ch,
                npm_timestamp_column_name=self.npm_timestamp_column_name,
                npm_split_events=self.npm_split_events,
            )
        return self._records

    @staticmethod
    def _event_rows(df: pd.DataFrame, record: EventStore) -> np.ndarray:
        """
        Select the rows of an event file that belong to one event store.

        Parameters
        ----------
        df : pd.DataFrame
            Contents of the event file.
        record : EventStore
            The store's record.

        Returns
        -------
        np.ndarray
            Integer row indices: every row when the file is not split, otherwise the rows
            whose value is the store's ``event_value``.
        """
        if record.event_value is None:
            return np.arange(df.shape[0])
        return np.flatnonzero(np.asarray(df.iloc[:, 1]) == record.event_value)

    def _channel_rows(self, df: pd.DataFrame, record: ChannelStore) -> np.ndarray:
        """
        Select the frames of a data file that belong to one photometry channel.

        In a file with a state column, a frame belongs to a wavelength when the wavelength's
        bit is set in its state word, whatever else is set alongside it. An opening
        initialization frame, which has every excitation bit set, belongs to none.

        Parameters
        ----------
        df : pd.DataFrame
            Contents of the data file.
        record : ChannelStore
            The channel's record.

        Returns
        -------
        np.ndarray
            Integer row indices: the frames its LED lit, or, in a file with no state column,
            every ``num_ch``-th frame from its interleave position.
        """
        if record.excitation_wavelength_in_nm is None:
            return np.arange(record.interleave_position, df.shape[0], self.num_ch)

        state_column = next(name for name in df.columns if name.lower() in STATE_COLUMN_NAMES)
        state = np.asarray(df[state_column], dtype=int)
        code = WAVELENGTH_TO_EXCITATION_CODE[record.excitation_wavelength_in_nm]
        is_lit = state & code == code
        if state[0] & EXCITATION_BITS == EXCITATION_BITS:
            is_lit[0] = False
        return np.flatnonzero(is_lit)

    @classmethod
    def _read_source_files(cls, folder_path: str | Path) -> dict[str, pd.DataFrame]:
        """
        Read every raw NPM file in a session folder.

        A CSV is an NPM file when :func:`_classify_csv_file` says so; Doric and standard-format
        CSVs sharing the folder belong to their own extractors. A file whose first row is numeric
        has no header, and its columns are labeled by position.

        Parameters
        ----------
        folder_path : str or Path
            Path to the session folder.

        Returns
        -------
        dict of str to pd.DataFrame
            Maps each raw file's name to its contents, sorted by name.
        """
        session_folder = Path(folder_path)
        derived_paths = {path for pattern in DERIVED_FILENAME_PATTERNS for path in session_folder.glob(pattern)}
        source_paths = [
            path
            for path in sorted(set(session_folder.glob("*.csv")) - derived_paths)
            if _classify_csv_file(str(path)) == "npm"
        ]
        dataframes = {}
        for path in source_paths:
            df = pd.read_csv(path, index_col=False)
            if not cls._has_text_header(df):
                df = pd.read_csv(path, header=None)
            dataframes[path.name] = df
        return dataframes

    @staticmethod
    def _has_text_header(df: pd.DataFrame) -> bool:
        """
        Check whether a file is in the headered layout rather than the legacy header-less one.

        A header is text: no column label parses as a number. A file read without a header has
        its columns labeled by position, which parse as numbers too.

        Parameters
        ----------
        df : pd.DataFrame
            File contents, read either with its first row as the header or with none.

        Returns
        -------
        bool
            ``True`` if the column labels are a text header.
        """
        return not any(_is_number(name) for name in df.columns)

    @staticmethod
    def _timestamp_column_names(df: pd.DataFrame) -> list[str]:
        """
        List a headered data file's timestamp columns.

        A timestamp column is one whose name contains ``"timestamp"``, case-insensitively.

        Parameters
        ----------
        df : pd.DataFrame
            Contents of a data file with a text header, as read by :meth:`_read_source_files`.

        Returns
        -------
        list of str
            Names of the timestamp columns, in file order.
        """
        return [name for name in df.columns if TIMESTAMP_COLUMN_SUBSTRING in name.lower()]

    @staticmethod
    def _is_event_file(df: pd.DataFrame) -> bool:
        """
        Check whether a file holds TTL events rather than photometry.

        An event file has two columns: a timestamp and a value per event.

        Parameters
        ----------
        df : pd.DataFrame
            File contents, as read by :meth:`_read_source_files`.

        Returns
        -------
        bool
            ``True`` if the file is an event file.
        """
        return len(df.columns) == 2

    @staticmethod
    def _has_state_column(df: pd.DataFrame) -> bool:
        """
        Check whether a headered data file has a ``Flags``/``LedState`` column.

        Parameters
        ----------
        df : pd.DataFrame
            Contents of a data file with a text header.

        Returns
        -------
        bool
            ``True`` if a column, matched case-insensitively, is the state column.
        """
        return any(name.lower() in STATE_COLUMN_NAMES for name in df.columns)

    @staticmethod
    def _is_region_column(name: str) -> bool:
        """
        Check whether a headered data file's column holds a region's fluorescence.

        A region is every column left after removing the blank-header columns, the clocks,
        the frame counters, the state word and the digital lines.

        Parameters
        ----------
        name : str
            Column name.

        Returns
        -------
        bool
            ``True`` if the column is a region.
        """
        if BLANK_HEADER_COLUMN_PATTERN.fullmatch(name):
            return False
        lowered_name = name.lower()
        if TIMESTAMP_COLUMN_SUBSTRING in lowered_name:
            return False
        if lowered_name.endswith("counter"):
            return False
        return lowered_name not in NON_REGION_COLUMN_NAMES

    @classmethod
    def _region_columns(cls, df: pd.DataFrame) -> list[str]:
        """
        List a headered data file's region columns.

        Parameters
        ----------
        df : pd.DataFrame
            Contents of a data file with a text header.

        Returns
        -------
        list of str
            Names of the region columns, in file order.
        """
        return [name for name in df.columns if cls._is_region_column(name)]

    @staticmethod
    def _excitation_wavelengths(df: pd.DataFrame) -> list[int]:
        """
        List the excitation wavelengths a file with a state column records.

        A recording can open with an initialization frame that has every excitation bit set;
        that first row is left out, so it does not count toward every wavelength.

        Parameters
        ----------
        df : pd.DataFrame
            Contents of a data file with a state column.

        Returns
        -------
        list of int
            Wavelengths, in nanometers and ascending, whose bit is set in any frame.
        """
        state_column = next(name for name in df.columns if name.lower() in STATE_COLUMN_NAMES)
        state = np.asarray(df[state_column], dtype=int)
        if state[0] & EXCITATION_BITS == EXCITATION_BITS:
            state = state[1:]
        state_values = np.unique(state)

        wavelengths = []
        for wavelength, code in sorted(WAVELENGTH_TO_EXCITATION_CODE.items()):
            if np.any(state_values & code == code):
                wavelengths.append(wavelength)
        return wavelengths

    @classmethod
    def _store_records(
        cls,
        dataframes: dict[str, pd.DataFrame],
        *,
        num_ch: int,
        npm_timestamp_column_name: str | None,
        npm_split_events: dict[str, bool] | None,
    ) -> dict[str, EventStore | ChannelStore]:
        """
        Work out every store the session yields and where each one comes from.

        This is the one place a store is defined: the store IDs, the provenance and the
        demultiplexing all read these records.

        Parameters
        ----------
        dataframes : dict of str to pd.DataFrame
            The session's raw files, as read by :meth:`_read_source_files`.
        num_ch : int
            Number of interleaved channels in a file with no state column.
        npm_timestamp_column_name : str or None
            Timestamp column to read in files that offer more than one; ``None`` reads the
            first.
        npm_split_events : dict of str to bool or None
            Maps each event file's name to whether it is split by value; ``None`` splits none.

        Returns
        -------
        dict of str to EventStore or ChannelStore
            Maps each store ID, in file order, to its record.

        Raises
        ------
        ValueError
            If a data file's state column sets no excitation bit, if a headered data file has
            no region column, or if its timestamp column cannot be resolved.
        """
        if npm_split_events is None:
            npm_split_events = {}

        store_records = {}
        for file_name, df in dataframes.items():
            if cls._is_event_file(df):
                split_events = npm_split_events.get(file_name, False)
                store_records.update(cls._event_store_records(file_name, df, split_events=split_events))
                continue

            timestamp_column = cls._resolve_timestamp_column(
                df, npm_timestamp_column_name=npm_timestamp_column_name, file_name=file_name
            )
            if cls._has_text_header(df) and cls._has_state_column(df):
                wavelengths = cls._excitation_wavelengths(df)
                if not wavelengths:
                    raise ValueError(
                        f"NPM file '{file_name}' has a state column whose values set no excitation bit, so "
                        f"it names no channel. The three lowest bits select the LED: "
                        f"{WAVELENGTH_TO_EXCITATION_CODE}."
                    )
                region_columns = cls._region_columns(df)
                if not region_columns:
                    raise ValueError(
                        f"NPM file '{file_name}' has no region columns. Found columns: {list(df.columns)}, "
                        "all of which name a clock, a frame counter, the state column, a digital line, or "
                        "nothing at all."
                    )
                store_records.update(
                    cls._excitation_store_records(
                        file_name, wavelengths, region_columns, timestamp_column=timestamp_column
                    )
                )
            else:
                if cls._has_text_header(df):
                    data_columns = cls._region_columns(df)
                    if not data_columns:
                        raise ValueError(
                            f"NPM file '{file_name}' has no region columns. Found columns: "
                            f"{list(df.columns)}, all of which name a clock, a frame counter, a digital "
                            "line, or nothing at all."
                        )
                else:
                    data_columns = list(df.columns[1:])
                store_records.update(
                    cls._stride_store_records(file_name, data_columns, num_ch=num_ch, timestamp_column=timestamp_column)
                )
        return store_records

    @classmethod
    def _resolve_timestamp_column(
        cls, df: pd.DataFrame, *, npm_timestamp_column_name: str | None, file_name: str
    ) -> str | int:
        """
        Pick the column a data file is timed by.

        A header-less file's timestamps are its leading column. A headered file offering a
        single timestamp column is timed by it whatever ``npm_timestamp_column_name`` names,
        so a session-wide choice may name a column some of its files lack.

        Parameters
        ----------
        df : pd.DataFrame
            Data file contents, as read by :meth:`_read_source_files`.
        npm_timestamp_column_name : str or None
            Timestamp column to read in a file offering more than one; ``None`` reads the
            first.
        file_name : str
            Name of the file, used in messages.

        Returns
        -------
        str or int
            Name of the timestamp column, or ``0`` for a header-less file.

        Raises
        ------
        ValueError
            If a headered file has no timestamp column, or offers several of which none is
            ``npm_timestamp_column_name``.
        """
        if not cls._has_text_header(df):
            return 0

        timestamp_column_names = cls._timestamp_column_names(df)
        if not timestamp_column_names:
            raise ValueError(
                f"NPM file '{file_name}' carries a text header but no timestamp column. "
                f"Found columns: {list(df.columns)}."
            )
        if len(timestamp_column_names) == 1 or npm_timestamp_column_name is None:
            return timestamp_column_names[0]
        if npm_timestamp_column_name not in timestamp_column_names:
            raise ValueError(
                f"Provided timestamp_column_name '{npm_timestamp_column_name}' not found in "
                f"columns {timestamp_column_names}."
            )
        return npm_timestamp_column_name

    @classmethod
    def _layout_flag(cls, df: pd.DataFrame) -> str:
        """
        Name a file's layout as :meth:`discover_events_and_flags` reports it.

        Parameters
        ----------
        df : pd.DataFrame
            File contents, as read by :meth:`_read_source_files`.

        Returns
        -------
        str
            ``"event_np"``, ``"data_np_v2"`` (a state column names each frame's LED) or
            ``"data_np"`` (it does not).
        """
        if cls._is_event_file(df):
            return "event_np"
        if cls._has_text_header(df) and cls._has_state_column(df):
            return "data_np_v2"
        return "data_np"

    @staticmethod
    def _event_store_records(file_name: str, df: pd.DataFrame, *, split_events: bool) -> dict[str, EventStore]:
        """
        Define an event file's stores.

        Parameters
        ----------
        file_name : str
            Name of the event file.
        df : pd.DataFrame
            Contents of the event file.
        split_events : bool
            Whether the file is split into one store per distinct value.

        Returns
        -------
        dict of str to EventStore
            When split, maps ``event{value}`` for each distinct value, in sorted order, to its
            record; otherwise maps ``"event0"`` to a record whose ``event_value`` is ``None``.
        """
        if not split_events:
            return {"event0": EventStore(file=file_name, event_value=None)}
        store_records = {}
        for value in np.unique(df.iloc[:, 1]):
            store_records[f"event{value}"] = EventStore(file=file_name, event_value=value)
        return store_records

    @staticmethod
    def _excitation_store_records(
        file_name: str, wavelengths: list[int], region_columns: list[str], *, timestamp_column: str
    ) -> dict[str, ChannelStore]:
        """
        Define the stores of a data file with a state column.

        Parameters
        ----------
        file_name : str
            Name of the data file.
        wavelengths : list of int
            Excitation wavelengths the file records, in nanometers.
        region_columns : list of str
            The file's region columns.
        timestamp_column : str
            Column the file is timed by.

        Returns
        -------
        dict of str to ChannelStore
            Maps ``{stem}_{wavelength}nm_{region}``, for each wavelength and then each region,
            to the channel's record.
        """
        stem = Path(file_name).stem
        store_records = {}
        for wavelength in wavelengths:
            for region_column in region_columns:
                store_records[f"{stem}_{wavelength}nm_{region_column}"] = ChannelStore(
                    file=file_name,
                    excitation_wavelength_in_nm=wavelength,
                    interleave_position=None,
                    data_column=region_column,
                    timestamp_column=timestamp_column,
                )
        return store_records

    @staticmethod
    def _stride_store_records(
        file_name: str, data_columns: list, *, num_ch: int, timestamp_column: str | int
    ) -> dict[str, ChannelStore]:
        """
        Define the stores of a data file with no state column.

        Parameters
        ----------
        file_name : str
            Name of the data file.
        data_columns : list
            The file's data columns.
        num_ch : int
            Number of channels interleaved in the file.
        timestamp_column : str or int
            Column the file is timed by.

        Returns
        -------
        dict of str to ChannelStore
            Maps ``{stem}_{slot}{n}``, for each cycle slot (``chev``, ``chod``, ``chpr``) and
            then each data column numbered from 1, to the channel's record.

        Raises
        ------
        ValueError
            If ``num_ch`` exceeds the three cycle slots.
        """
        if num_ch > len(STRIDE_CHANNEL_SLOTS):
            raise ValueError(
                f"Number of channels is set to {num_ch}, which exceeds the maximum of "
                f"{len(STRIDE_CHANNEL_SLOTS)} channels supported for an NPM file with no state column. "
                f"Set 'Number of channels' to {len(STRIDE_CHANNEL_SLOTS)} or fewer on the Label Stores page."
            )
        stem = Path(file_name).stem
        store_records = {}
        for slot_index in range(num_ch):
            slot = STRIDE_CHANNEL_SLOTS[slot_index]
            for column_number, data_column in enumerate(data_columns, start=1):
                store_records[f"{stem}_{slot}{column_number}"] = ChannelStore(
                    file=file_name,
                    excitation_wavelength_in_nm=None,
                    interleave_position=slot_index,
                    data_column=data_column,
                    timestamp_column=timestamp_column,
                )
        return store_records
