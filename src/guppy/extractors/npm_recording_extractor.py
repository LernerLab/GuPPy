import logging
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import panel as pn

from guppy.extractors import CsvRecordingExtractor
from guppy.extractors.detect_acquisition_formats import (
    _classify_csv_file,
    _is_event_csv,
)

pn.extension()

logger = logging.getLogger(__name__)

# Timestamp units an NPM session can be recorded in, and the factor that converts them
# to seconds. Nothing in the raw files states which one applies, so it is a parameter.
TIME_UNIT_DIVISORS = {"seconds": 1.0, "milliseconds": 1e3, "microseconds": 1e6}
DEFAULT_TIME_UNIT = "seconds"
# Channel count assumed until the Label Stores page asks for one.
DEFAULT_NUM_CHANNELS = 2
# pandas names a column whose header cell is blank ``Unnamed: {position}``. Such a column
# states no channel name, so it is not one of the file's data channels.
_UNNAMED_COLUMN_PATTERN = re.compile(r"Unnamed: \d+")

# The three lowest bits of a ``Flags``/``LedState`` word are one flag per excitation LED; the
# higher bits are digital TTL lines. A wavelength's rows are those whose word has that
# wavelength's bit set, whatever else is set alongside it. Two consequences follow:
# rows sharing an excitation LED but differing in a TTL line carry different values (17 and 273
# are both 415 nm), and a rig can strobe two LEDs in one frame (6 is 470 nm and 560 nm together,
# their emission landing in different region columns), so that frame belongs to both channels.
EXCITATION_BITS = 0b111
WAVELENGTH_TO_EXCITATION_CODE = {415: 1, 470: 2, 560: 4}

# Every column an NPM file writes that is not a region: the clock and frame index, the
# excitation/TTL word, and the digital lines. The set is closed, whereas the region names are
# not -- the prefix is user text when a GroupRegions operator is upstream, and older packages
# spelled it "Channel" rather than "Region" -- so a region is recognized by subtracting these
# rather than by matching a name pattern. ``Stimulation``, ``Output0/1`` and ``Input0/1`` appear
# only in files written between 2021-05-20 and 2021-11-07.
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
# A variant we have not seen could add another clock or frame counter, so anything spelled like
# one is excluded too rather than read as a region. A clock is matched anywhere in the name,
# which is also how the timestamp columns themselves are found: the two rules have to agree, or
# a timestamp column the session did not choose would be demultiplexed as a region.
TIMESTAMP_COLUMN_SUBSTRING = "timestamp"
NON_REGION_COLUMN_SUFFIXES = ("counter",)
# Names of the interleaved channel slots of a header-less file, in cycle order.
STRIDE_CHANNEL_SLOTS = ("chev", "chod", "chpr")

# How a column of an NPM file is addressed: by its header name, or by its position in a file
# that has no header.
ColumnLabel = str | int


class NpmRecordingExtractor(CsvRecordingExtractor):
    """
    Extractor for fiber photometry data from Neurophotometrics (NPM) systems.

    NPM files store interleaved channels (and, optionally, a multi-type event
    column) in multi-column CSVs. This extractor demultiplexes those raw files
    into per-channel and per-event streams **entirely in memory** via
    :meth:`decompose`; ``read`` and ``count_samples`` are thin wrappers over it.
    Nothing is written to the source folder — only the final HDF5 outputs are
    written, by the inherited :meth:`CsvRecordingExtractor.save`.

    A channel is named ``{source file stem}_{wavelength}nm_{region column}`` when the file
    carries a ``Flags``/``LedState`` column naming the LED that lit each frame, and
    ``{source file stem}_ch{ev|od|pr}{n}`` when it does not and the channels have to be taken
    to cycle by row position.

    Parameters
    ----------
    folder_path : str
        Path to the session folder containing the NPM CSV files.
    num_ch : int, optional
        Number of interleaved channels expected. Default is 2.
    npm_timestamp_column_name : str, optional
        Timestamp column to use in files that have more than one. ``None`` lets
        the extractor pick the first timestamp column.
    npm_time_unit : str, optional
        Unit of the session's timestamps (``"seconds"``, ``"milliseconds"``, or
        ``"microseconds"``), applied to every file in the folder. ``None``
        defaults to seconds.
    npm_split_events : list of bool, optional
        Per-file flag controlling whether a multi-type event column is split
        into one event stream per unique value. ``None`` defaults to no split.
    """

    def __init__(
        self,
        folder_path: str,
        *,
        num_ch: int = 2,
        npm_timestamp_column_name: str | None = None,
        npm_time_unit: str | None = None,
        npm_split_events: list[bool] | None = None,
    ) -> None:
        self.folder_path = folder_path
        self.num_ch = num_ch
        self.npm_timestamp_column_name = npm_timestamp_column_name
        self.npm_time_unit = npm_time_unit
        self.npm_split_events = npm_split_events
        self._decomposed: dict[str, dict[str, np.ndarray]] | None = None
        self._store_provenance: dict[str, dict[str, object]] = {}

    @classmethod
    def discover_events_and_flags(
        cls, folder_path: str, num_ch: int, inputParameters: dict[str, object] | None
    ) -> tuple[list[str], list[str]]:
        """
        Discover available events and format flags from NPM files.

        This is read-only: it decomposes the raw NPM files in memory to
        enumerate the derived stream names, but writes nothing to disk.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing NPM files.
        num_ch : int
            Number of channels in the recording.
        inputParameters : dict, optional
            Input parameters containing NPM-specific configuration.

        Returns
        -------
        events : list of str
            Names of all events/stores available in the dataset.
        flags : list of str
            Format indicators or file type flags.
        """
        npm_timestamp_column_name = None
        npm_time_unit = None
        npm_split_events = None
        if isinstance(inputParameters, dict):
            npm_timestamp_column_name = inputParameters.get("npm_timestamp_column_name")
            npm_time_unit = inputParameters.get("npm_time_unit")
            # TODO: come up with a better name for npm_split_events that can be appropriately pluralized for a list
            npm_split_events = inputParameters.get("npm_split_events")

        streams, flags, _ = cls._decompose_streams(
            folder_path=folder_path,
            num_ch=num_ch,
            npm_timestamp_column_name=npm_timestamp_column_name,
            npm_time_unit=npm_time_unit,
            npm_split_events=npm_split_events,
        )
        return list(streams.keys()), flags

    @staticmethod
    def _list_npm_files(folder_path: str | Path) -> list[Path]:
        """List the raw NPM source files in a session folder, in processing order.

        Excludes the derived per-channel/per-event filenames and the
        single-column timestamp CSVs handled by
        :class:`~guppy.extractors.CsvRecordingExtractor`.

        Parameters
        ----------
        folder_path : str
            Path to the session folder.

        Returns
        -------
        list of Path
            Sorted paths of the raw NPM files. A path's stem is the prefix its channels are
            named with, and its index in this list is the index its ``npm_split_events``
            entry is read from.
        """
        session_folder = Path(folder_path)
        path = sorted(session_folder.glob("*.csv")) + sorted(session_folder.glob("*.doric"))
        path_chev = list(session_folder.glob("*chev*"))
        path_chod = list(session_folder.glob("*chod*"))
        path_chpr = list(session_folder.glob("*chpr*"))
        path_event = list(session_folder.glob("event*"))
        path_chev_chod_event = path_chev + path_chod + path_event + path_chpr

        path = sorted(set(path) - set(path_chev_chod_event))
        return [csv_path for csv_path in path if not (csv_path.suffix == ".csv" and _is_event_csv(csv_path))]

    @classmethod
    def _classify_npm_file(cls, path: str | Path) -> tuple[str, pd.DataFrame, bool]:
        """Read one raw NPM file and determine which NPM layout it uses.

        Parameters
        ----------
        path : str or Path
            Path to a raw NPM file, as returned by :meth:`_list_npm_files`.

        Returns
        -------
        flag : str
            ``"data_np"`` (interleaved channels, no state column),
            ``"data_np_v2"`` (interleaved channels annotated by a
            ``Flags``/``LedState`` column), or ``"event_np"`` (timestamp/value
            event pairs).
        dataframe : pd.DataFrame
            The file's contents, read with or without a text header according to
            the layout.
        columns_are_strings : bool
            Whether the file carries a text header row.
        """
        extension = Path(path).name.split(".")[-1]
        if extension == "doric":
            raise ValueError(f"Doric files are not supported by NpmRecordingExtractor; got '{path}'.")
        df = pd.read_csv(path, header=None, nrows=2, index_col=False, dtype=str)
        df = df.dropna(axis=1, how="all")
        header_values = np.array(df).flatten()
        check_all_str = []
        for element in header_values:
            try:
                float(element)
            except:
                check_all_str.append(element)
        if len(check_all_str) == len(header_values):
            raise ValueError(
                f"CSV file '{path}' appears to be a Doric .csv (all-string header rows). "
                "NpmRecordingExtractor only supports NPM .csv files; use the Doric extractor instead."
            )
        df = pd.read_csv(path, index_col=False)
        _, numeric_headers = cls._check_header(df)

        # check dataframe structure and read data accordingly
        if len(numeric_headers) > 0:
            columns_are_strings = False
            df = pd.read_csv(path, header=None)
        else:
            columns_are_strings = True
        columns = np.array(list(df.columns), dtype=str)

        # check the structure of dataframe and assign flag to the type of file
        if len(columns) == 1:
            raise ValueError(
                f"CSV file '{path}' has 1 column (event .csv layout). "
                "NpmRecordingExtractor only supports NPM .csv files; use the standard CSV extractor for event timestamp files."
            )
        if len(columns) == 3:
            raise ValueError(
                f"CSV file '{path}' has 3 columns {list(columns)} (data .csv layout). "
                "NpmRecordingExtractor only supports NPM .csv files; use the standard CSV extractor for 3-column data files."
            )
        if len(columns) == 2:
            flag = "event_or_data_np"
        else:
            flag = "data_np"

        if columns_are_strings and (
            "flags" in np.char.lower(np.array(columns)) or "ledstate" in np.char.lower(np.array(columns))
        ):
            flag = flag + "_v2"

        if flag == "event_or_data_np":
            second_column_values = list(df.iloc[:, 1])
            check_float = [True for value in second_column_values if isinstance(value, float)]
            if len(second_column_values) == len(check_float) and not columns_are_strings:
                flag = "data_np"
            else:
                flag = "event_np"

        return flag, df, columns_are_strings

    @classmethod
    def _decompose_streams(
        cls,
        *,
        folder_path: str,
        num_ch: int,
        npm_timestamp_column_name: str | None,
        npm_time_unit: str | None,
        npm_split_events: list[bool] | None,
    ) -> tuple[dict[str, dict[str, np.ndarray]], list[str], dict[str, dict[str, object]]]:
        """
        Demultiplex raw NPM files into per-channel and per-event streams in memory.

        Pure function: reads the raw NPM CSVs and returns the derived streams,
        keyed by event name, without writing anything to disk.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing the raw NPM files.
        num_ch : int
            Number of interleaved channels expected.
        npm_timestamp_column_name : str or None
            Timestamp column to use in files with more than one; ``None`` for
            default selection.
        npm_time_unit : str or None
            Unit of the session's timestamps, applied to every file in the
            folder; ``None`` for seconds.
        npm_split_events : list of bool or None
            Per-file event-split flags; ``None`` for no split.

        Returns
        -------
        streams : dict
            Maps event name to a stream dict. Data channels
            (``{stem}_{wavelength}nm_{region}``, or ``{stem}_chev{j}`` / ``chod{j}`` /
            ``chpr{j}`` for a file that does not name the LED lighting each frame) carry
            ``timestamps``, ``data``, and ``sampling_rate``; event streams
            (``event{value}`` / ``event0``) carry only ``timestamps``.
        flags : list of str
            One format flag per raw source file processed.
        store_provenance : dict
            Maps each data channel's name to the source file, excitation (or interleave
            position) and column it was demultiplexed from. Event streams are absent: they are
            read whole from their own file and need no such record.
        """
        logger.debug("If it exists, importing NPM file based on the structure of file")
        divisor = cls._time_unit_divisor(npm_time_unit)
        path = cls._list_npm_files(folder_path)

        streams: dict[str, dict[str, np.ndarray]] = {}
        # Track derived stream names per channel group, in creation order, so the cross-file
        # pairing (the later groups borrow the first group's timebase) is deterministic. A
        # group is one channel slot of one file, and the slots are ordered the same way in
        # every file of a session: by excitation wavelength, or by cycle position.
        channel_group_names: list[list[str]] = [[] for _ in STRIDE_CHANNEL_SLOTS]
        # What each derived channel was read from, recorded so a consumer of the run folder can
        # resolve a store back to its source without re-deriving this demultiplexing.
        store_provenance: dict[str, dict[str, object]] = {}
        flags: list[str] = []
        for i, file_path in enumerate(path):
            # TODO: validate npm_split_events length
            if npm_split_events is None:
                split_events = False
            else:
                split_events = npm_split_events[i]

            flag, df, columns_are_strings = cls._classify_npm_file(file_path)

            flags.append(flag)
            logger.info(flag)
            if flag == "event_np":
                type_val = np.array(df.iloc[:, 1])
                type_val_unique = np.unique(type_val)
                if split_events:
                    timestamps = np.array(df.iloc[:, 0])
                    for j in range(len(type_val_unique)):
                        matching_indices = np.where(type_val == type_val_unique[j])
                        name = "event" + str(type_val_unique[j])
                        streams[name] = {"timestamps": np.asarray(timestamps[matching_indices], dtype=float)}
                else:
                    timestamps = np.array(df.iloc[:, 0])
                    name = "event" + str(0)
                    streams[name] = {"timestamps": np.asarray(timestamps, dtype=float)}
                continue

            name_prefix = f"{file_path.stem}_"
            timestamp_column, data_columns = cls._resolve_columns(
                df,
                columns_are_strings=columns_are_strings,
                timestamp_column_name=npm_timestamp_column_name,
                source_path=file_path,
            )
            if flag == "data_np_v2":
                channel_groups, file_provenance = cls._decompose_by_excitation(
                    df,
                    name_prefix=name_prefix,
                    timestamp_column=timestamp_column,
                    data_columns=data_columns,
                    source_path=file_path,
                )
            else:
                channel_groups, file_provenance = cls._decompose_by_stride(
                    df,
                    name_prefix=name_prefix,
                    timestamp_column=timestamp_column,
                    data_columns=data_columns,
                    num_ch=num_ch,
                    source_path=file_path,
                )
            store_provenance.update(file_provenance)
            for group_index, channel_group in enumerate(channel_groups):
                for name, stream in channel_group:
                    streams[name] = stream
                    channel_group_names[group_index].append(name)

        # Convert every stream to seconds with the session's single timestamp unit, then
        # compute sampling rates. Timestamps keep the acquisition's own clock.
        for stream in streams.values():
            stream["timestamps"] = stream["timestamps"] / divisor

        populated_groups = [names for names in channel_group_names if len(names) > 0]
        if len({len(names) for names in populated_groups}) > 1:
            channel_group_counts = {names[0]: len(names) for names in populated_groups}
            message = (
                "Number of channel files must match across channel groups. Found per-channel-group "
                f"counts, keyed by each group's first channel: {channel_group_counts}."
            )
            logger.error(message)
            raise ValueError(message)

        # An interleaved recording lights the whole LED cycle within one frame period, and the
        # control fit downstream reads the channels as simultaneous samples, so every channel of
        # a file is stamped with the first slot's clock and trimmed to the length they share.
        reference_names = channel_group_names[0]
        paired_groups = [names for names in channel_group_names[1:] if len(names) > 0]
        for j in range(len(reference_names)):
            reference_stream = streams[reference_names[j]]
            reference_timestamps = reference_stream["timestamps"]
            sampling_rate = reference_timestamps.shape[0] / (reference_timestamps[-1] - reference_timestamps[0])
            reference_stream["sampling_rate"] = np.array([sampling_rate])

            for paired_names in paired_groups:
                paired_stream = streams[paired_names[j]]
                sample_count = min(reference_timestamps.shape[0], paired_stream["data"].shape[0])
                paired_stream["timestamps"] = reference_timestamps[:sample_count]
                paired_stream["data"] = paired_stream["data"][:sample_count]
                paired_stream["sampling_rate"] = np.array([sampling_rate])

        logger.info("Importing of NPM file is done.")
        return streams, flags, store_provenance

    @classmethod
    def _resolve_columns(
        cls,
        df: pd.DataFrame,
        *,
        columns_are_strings: bool,
        timestamp_column_name: str | None,
        source_path: str | Path,
    ) -> tuple[ColumnLabel, list[ColumnLabel]]:
        """Pick a data file's timestamp column and its data columns.

        Parameters
        ----------
        df : pd.DataFrame
            NPM data file contents.
        columns_are_strings : bool
            Whether the file carries a text header, as reported by
            :meth:`_classify_npm_file`.
        timestamp_column_name : str or None
            Timestamp column to read; ``None`` reads the first one. A file offering a single
            timestamp column ignores it, so a session-wide choice may name a column that some
            of the session's files do not have.
        source_path : str or Path
            Path the frame was read from, used to name the file in messages.

        Returns
        -------
        timestamp_column : str or int
            Label of the column holding the timestamps.
        data_columns : list
            Labels of the columns holding channel data, in file order.
        """
        if not columns_are_strings:
            # Nothing names these columns: the timestamps lead and the rest are data.
            return df.columns[0], list(df.columns[1:])

        timestamp_column_names = [name for name in df.columns if TIMESTAMP_COLUMN_SUBSTRING in str(name).lower()]
        if not timestamp_column_names:
            message = (
                f"NPM file '{source_path}' carries a text header but no timestamp column. "
                f"Found columns: {list(df.columns)}."
            )
            logger.error(message)
            raise ValueError(message)
        if (
            len(timestamp_column_names) > 1
            and timestamp_column_name is not None
            and timestamp_column_name not in timestamp_column_names
        ):
            message = (
                f"Provided timestamp_column_name '{timestamp_column_name}' not found in "
                f"columns {timestamp_column_names}."
            )
            logger.error(message)
            raise ValueError(message)
        timestamp_column = (
            timestamp_column_name if timestamp_column_name in timestamp_column_names else timestamp_column_names[0]
        )

        blank_header_column_names = [name for name in df.columns if _UNNAMED_COLUMN_PATTERN.fullmatch(str(name))]
        if blank_header_column_names:
            logger.warning(
                "NPM file '%s' has blank-header columns %s; they name no channel and are excluded "
                "from the derived streams.",
                source_path,
                blank_header_column_names,
            )
        data_columns = [name for name in df.columns if cls._is_region_column(name)]
        if not data_columns:
            message = (
                f"NPM file '{source_path}' has no region columns. Found columns: {list(df.columns)}, "
                "all of which name a clock, a frame counter, the channel-state word, a digital line, "
                "or nothing at all."
            )
            logger.error(message)
            raise ValueError(message)
        return timestamp_column, data_columns

    @staticmethod
    def _is_region_column(name: object) -> bool:
        """Whether a column name is a region rather than one of the columns NPM writes around them.

        Matched on the lowercased name, since :meth:`_detect_state_column` accepts case variants
        of ``Flags``/``LedState`` and a case-sensitive test here would read the state column as a
        region.
        """
        if _UNNAMED_COLUMN_PATTERN.fullmatch(str(name)):
            return False
        lowered = str(name).lower()
        if TIMESTAMP_COLUMN_SUBSTRING in lowered:
            return False
        return lowered not in NON_REGION_COLUMN_NAMES and not lowered.endswith(NON_REGION_COLUMN_SUFFIXES)

    @classmethod
    def _detect_state_column(cls, df: pd.DataFrame, source_path: str | Path) -> ColumnLabel:
        """Return the file's channel-state column, i.e. its ``Flags`` or ``LedState`` column."""
        column_by_name = cls._column_by_lowercase_name(df)
        for candidate in ("flags", "ledstate"):
            if candidate in column_by_name:
                return column_by_name[candidate]
        message = (
            "File type indicates Neurophotometrics newer version data but the columns do not "
            f"contain a 'Flags' or 'LedState' column. Found columns: {list(df.columns)}."
        )
        logger.error(message)
        raise ValueError(message)

    @staticmethod
    def _startup_row_count(state: np.ndarray) -> int:
        """Return how many leading rows of ``state`` are an initialization frame.

        A recording can open with a frame that is not a measurement: it is written with every
        excitation bit set (``7``, or ``23`` when a digital output is high alongside) while the
        frame itself is dark. It would otherwise land in all three wavelengths at once, since it
        carries all three bits. Only the first row is tested, so a genuine
        simultaneous-excitation frame later in the recording is kept and reaches every channel
        it belongs to. A startup frame coded ``0`` or ``16`` needs no test: with no excitation
        bit set it matches no wavelength and drops out of every channel on its own.
        """
        return 1 if int(state[0]) & EXCITATION_BITS == EXCITATION_BITS else 0

    @classmethod
    def _decompose_by_excitation(
        cls,
        df: pd.DataFrame,
        *,
        name_prefix: str,
        timestamp_column: ColumnLabel,
        data_columns: list[ColumnLabel],
        source_path: str | Path,
    ) -> tuple[list[list[tuple[str, dict[str, np.ndarray]]]], dict[str, dict[str, object]]]:
        """Split a file annotated by a ``Flags``/``LedState`` column into per-excitation channels.

        Parameters
        ----------
        df : pd.DataFrame
            NPM data file contents, carrying a state column.
        name_prefix : str
            Prefix the derived stream names are built on (e.g. ``"signals_"``).
        timestamp_column : str or int
            Label of the column holding the timestamps.
        data_columns : list
            Labels of the region columns to read.
        source_path : str or Path
            Path the frame was read from, used to name the file in messages.

        Returns
        -------
        channel_groups : list of list of (str, dict)
            One list per excitation wavelength, in ascending wavelength order; within it, one
            ``(name, stream)`` pair per region, in file order.
        store_provenance : dict
            Maps each derived name to the file, excitation and column it was read from.
        """
        state_column = cls._detect_state_column(df, source_path)
        state = np.asarray(df[state_column], dtype=int)
        startup_row_count = cls._startup_row_count(state)
        state_values = np.unique(state[startup_row_count:])

        timestamps = np.asarray(df[timestamp_column], dtype=float)
        channel_groups = []
        store_provenance: dict[str, dict[str, object]] = {}
        for wavelength, code in sorted(WAVELENGTH_TO_EXCITATION_CODE.items()):
            # ``value & code == code`` asks whether this wavelength's bit is set in the row's
            # word, ignoring whatever else is set alongside it -- which is what lets one channel
            # span several words (415 nm as both 17 and 273) and one word reach several channels
            # (6 is claimed by 470 nm and by 560 nm).
            matching_states = state_values[(state_values & code) == code]
            if matching_states.size == 0:
                continue
            rows = np.zeros(state.shape[0], dtype=bool)
            rows[startup_row_count:] = np.isin(state[startup_row_count:], matching_states)
            channel_group = []
            for column in data_columns:
                name = f"{name_prefix}{wavelength}nm_{column}"
                channel_group.append(
                    (
                        name,
                        {
                            "timestamps": timestamps[rows],
                            "data": np.asarray(df[column], dtype=float)[rows],
                        },
                    )
                )
                store_provenance[name] = {
                    "file": Path(source_path).name,
                    "excitation_wavelength_in_nm": wavelength,
                    "data_column": column,
                }
            channel_groups.append(channel_group)

        if not channel_groups:
            message = (
                f"NPM file '{source_path}' has a '{state_column}' column whose values "
                f"{state_values.tolist()} set no excitation bit, so it names no channel. The three "
                f"lowest bits select the LED: {WAVELENGTH_TO_EXCITATION_CODE}."
            )
            logger.error(message)
            raise ValueError(message)
        return channel_groups, store_provenance

    @classmethod
    def _decompose_by_stride(
        cls,
        df: pd.DataFrame,
        *,
        name_prefix: str,
        timestamp_column: ColumnLabel,
        data_columns: list[ColumnLabel],
        num_ch: int,
        source_path: str | Path,
    ) -> tuple[list[list[tuple[str, dict[str, np.ndarray]]]], dict[str, dict[str, object]]]:
        """Split a file with no state column into channels by row position.

        Nothing in such a file says which LED lit a frame, so the channels are taken to cycle in
        a fixed order and ``num_ch`` is whatever the Input Parameters GUI was set to.

        Parameters
        ----------
        df : pd.DataFrame
            NPM data file contents.
        name_prefix : str
            Prefix the derived stream names are built on.
        timestamp_column : str or int
            Label of the column holding the timestamps.
        data_columns : list
            Labels of the data columns to read.
        num_ch : int
            Number of interleaved channels expected.
        source_path : str or Path
            Path the frame was read from, used to name the file in the provenance.

        Returns
        -------
        channel_groups : list of list of (str, dict)
            One list per channel slot, in cycle order; within it, one ``(name, stream)`` pair
            per data column, in file order.
        store_provenance : dict
            Maps each derived name to the file, cycle position and column it was read from.
        """
        if num_ch > len(STRIDE_CHANNEL_SLOTS):
            message = (
                f"Number of channels in the Input Parameters GUI is set to {num_ch}, which exceeds the "
                f"maximum of {len(STRIDE_CHANNEL_SLOTS)} channels supported for NPM files. Set "
                f"'Number of channels' to {len(STRIDE_CHANNEL_SLOTS)} or fewer in the Input Parameters GUI."
            )
            logger.error(message)
            raise ValueError(message)

        timestamps = np.asarray(df[timestamp_column], dtype=float)
        channel_groups = []
        store_provenance: dict[str, dict[str, object]] = {}
        for slot_index in range(num_ch):
            rows = np.arange(slot_index, df.shape[0], num_ch)
            channel_group = []
            for column_index, column in enumerate(data_columns):
                name = f"{name_prefix}{STRIDE_CHANNEL_SLOTS[slot_index]}{column_index + 1}"
                channel_group.append(
                    (
                        name,
                        {
                            "timestamps": timestamps[rows],
                            "data": np.asarray(df[column], dtype=float)[rows],
                        },
                    )
                )
                store_provenance[name] = {
                    "file": Path(source_path).name,
                    "excitation_wavelength_in_nm": None,
                    "interleave_position": slot_index,
                    "data_column": column,
                }
            channel_groups.append(channel_group)
        return channel_groups, store_provenance

    @staticmethod
    def _time_unit_divisor(npm_time_unit: str | None) -> float:
        """Return the factor converting ``npm_time_unit`` timestamps to seconds.

        Parameters
        ----------
        npm_time_unit : str or None
            One of the keys of :data:`TIME_UNIT_DIVISORS`. ``None`` selects
            :data:`DEFAULT_TIME_UNIT`.

        Returns
        -------
        float
            Divisor to apply to raw timestamps.
        """
        time_unit = npm_time_unit if npm_time_unit is not None else DEFAULT_TIME_UNIT
        if time_unit not in TIME_UNIT_DIVISORS:
            message = (
                f"npm_time_unit='{time_unit}' is not a recognized timestamp unit; "
                f"choose one of {list(TIME_UNIT_DIVISORS)}."
            )
            logger.error(message)
            raise ValueError(message)
        return TIME_UNIT_DIVISORS[time_unit]

    def decompose(self) -> dict[str, dict[str, np.ndarray]]:
        """
        Demultiplex this session's raw NPM files into in-memory streams.

        Returns
        -------
        dict
            Maps event name to a stream dict. Data channels carry
            ``timestamps``, ``data``, and ``sampling_rate``; event streams carry
            only ``timestamps``. The result is cached on the instance.
        """
        if self._decomposed is None:
            streams, _, store_provenance = self._decompose_streams(
                folder_path=self.folder_path,
                num_ch=self.num_ch,
                npm_timestamp_column_name=self.npm_timestamp_column_name,
                npm_time_unit=self.npm_time_unit,
                npm_split_events=self.npm_split_events,
            )
            self._decomposed = streams
            self._store_provenance = store_provenance
        return self._decomposed

    def store_provenance(self) -> dict[str, dict[str, object]]:
        """Return what each data channel of this session was demultiplexed from.

        NPM store names are invented here — no column of the raw file carries one — so a run
        folder that records only the names leaves a reader to re-derive this demultiplexing from
        them. This is that record: for each channel, the source file, the excitation wavelength
        that lit it (``None`` where the file names no LED, with the cycle position instead), and
        the column it was read from. It is written to ``.npm_params.json`` beside
        ``storesList.csv`` so a consumer of the run folder can resolve a store without
        reproducing any of this module's arithmetic.

        Returns
        -------
        dict
            Maps each data channel's store name to its source record. Event streams are absent:
            they are read whole from their own file and need no such record.
        """
        self.decompose()
        return self._store_provenance

    @classmethod
    def _timestamp_column_spans(cls, folder_path: str) -> dict[str, tuple[float, float]]:
        """Return each named timestamp column's raw span across the session's data files.

        Raw means in the file's own units, before ``npm_time_unit`` is applied. Header-less
        sessions name no columns and yield an empty mapping.

        Parameters
        ----------
        folder_path : str
            Path to the session folder.

        Returns
        -------
        dict
            Maps timestamp column name to its ``(minimum, maximum)`` raw value.
        """
        column_spans: dict[str, tuple[float, float]] = {}
        for path in cls._list_npm_files(folder_path):
            flag, df, columns_are_strings = cls._classify_npm_file(path)
            if flag == "event_np" or not columns_are_strings:
                continue
            for name in df.columns:
                if "timestamp" not in str(name).lower():
                    continue
                values = np.asarray(df[name], dtype=float)
                span = (float(np.nanmin(values)), float(np.nanmax(values)))
                if str(name) in column_spans:
                    previous = column_spans[str(name)]
                    span = (min(previous[0], span[0]), max(previous[1], span[1]))
                column_spans[str(name)] = span
        return column_spans

    def _validate_events_share_data_clock(self, *, events: list[str], streams: dict) -> None:
        """Check that each selected event stream lies on the photometry channels' clock.

        A session's photometry file may offer several timestamp columns on different clocks,
        and ``npm_timestamp_column_name`` chooses between them for the photometry channels
        only — an NPM event file carries a single column, always on the acquisition's absolute
        clock. Selecting a column that is not that clock decouples the two streams, which stays
        invisible until PSTH computation indexes far outside the trace.

        Parameters
        ----------
        events : list of str
            The event names being read.
        streams : dict
            The decomposed streams, as returned by :meth:`decompose`.

        Raises
        ------
        ValueError
            If a selected event stream has no timestamp inside the photometry timespan.
        """
        # Only the photometry being read alongside these events is a valid reference: Step 2 batches
        # all of one extractor's stores into a single read, so a session whose traces come from
        # another format selects no NPM channel here and has no NPM clock to be checked against.
        data_timestamps = [streams[event]["timestamps"] for event in events if "data" in streams[event]]
        selected_event_names = [event for event in events if "data" not in streams[event]]
        if not data_timestamps or not selected_event_names:
            return

        data_start = min(float(timestamps[0]) for timestamps in data_timestamps)
        data_end = max(float(timestamps[-1]) for timestamps in data_timestamps)

        for event in selected_event_names:
            event_timestamps = streams[event]["timestamps"]
            if ((event_timestamps >= data_start) & (event_timestamps <= data_end)).any():
                continue

            message = (
                f"Event store '{event}' spans [{float(event_timestamps[0]):.4g}, "
                f"{float(event_timestamps[-1]):.4g}]s, which lies entirely outside the photometry "
                f"timespan [{data_start:.4g}, {data_end:.4g}]s, so no PSTH trial can be built from it. "
                f"{self._clock_advice(event_timestamps=event_timestamps)}"
            )
            logger.error(message)
            raise ValueError(message)

    def _clock_advice(self, *, event_timestamps: np.ndarray) -> str:
        """Build the fix clause naming the timestamp column the events actually ride."""
        divisor = self._time_unit_divisor(self.npm_time_unit)
        raw_event_timestamps = np.asarray(event_timestamps, dtype=float) * divisor
        column_spans = self._timestamp_column_spans(self.folder_path)

        for column_name, (column_start, column_end) in column_spans.items():
            if ((raw_event_timestamps >= column_start) & (raw_event_timestamps <= column_end)).any():
                return (
                    "NPM event files are written on the acquisition's absolute clock, which for this "
                    f"session is the '{column_name}' column (raw span [{column_start:.6g}, {column_end:.6g}]). "
                    f"Set Timestamp column to '{column_name}', and set Time unit to that column's own unit, "
                    "in the Label Stores NPM configuration."
                )

        if column_spans:
            offered = ", ".join(f"'{name}' [{start:.6g}, {end:.6g}]" for name, (start, end) in column_spans.items())
            return (
                "NPM event files are written on the acquisition's absolute clock, but none of this "
                f"session's timestamp columns ({offered}, raw) contains these event times. Check the "
                "Timestamp column and Time unit in the Label Stores NPM configuration."
            )

        return (
            "NPM event files are written on the acquisition's absolute clock. Check the Time unit in "
            "the Label Stores NPM configuration."
        )

    def read(self, *, events: list[str], outputPath: str) -> list[dict[str, Any]]:
        """
        Read data for the specified events from the in-memory decomposition.

        Parameters
        ----------
        events : list of str
            Event names to read. Each must be a key produced by
            :meth:`decompose`.
        outputPath : str
            Path to the output directory (unused by this extractor; required by
            the base-class interface).

        Returns
        -------
        list of dict
            One dictionary per event. Data channels produce dicts with keys
            ``store_id``, ``timestamps``, ``data``, and ``sampling_rate``;
            event streams produce dicts with keys ``store_id`` and
            ``timestamps``.

        Raises
        ------
        ValueError
            If a selected event stream lies entirely outside the photometry timespan.
        """
        streams = self.decompose()
        self._validate_events_share_data_clock(events=events, streams=streams)
        output_dicts = []
        for event in events:
            output_dicts.append({"store_id": event, **streams[event]})
        return output_dicts

    def count_samples(self, *, event: str) -> int:
        """Return the number of samples for ``event`` from the in-memory decomposition."""
        streams = self.decompose()
        if event not in streams:
            return 0
        stream = streams[event]
        if "data" in stream:
            return len(stream["data"])
        return len(stream["timestamps"])

    def stub(self, *, folder_path: str | Path, duration_in_seconds: float = 1.0) -> None:
        """
        Create a stubbed copy of the NPM folder with truncated signal files.

        Copies the folder to ``folder_path``, then truncates each raw NPM CSV
        (multi-column signal files and 2-column event/stimuli files) to
        approximately ``duration_in_seconds``. The cutoff timestamp is computed
        as the first value in the timestamp column plus ``duration_in_seconds``
        (scaled to milliseconds when the first timestamp value exceeds ``1e6``).

        Parameters
        ----------
        folder_path : str or Path
            Destination directory. Created if absent; overwritten if present.
        duration_in_seconds : float, optional
            Approximate signal duration to retain in seconds. Default is 1.0.
        """
        folder_path = Path(folder_path)
        if folder_path.exists():
            shutil.rmtree(folder_path)
        shutil.copytree(self.folder_path, folder_path)

        for csv_path in sorted(folder_path.glob("*.csv")):
            if _classify_csv_file(str(csv_path)) != "npm":
                continue
            df_probe = pd.read_csv(csv_path, index_col=False)
            _, float_conversions = self._check_header(df_probe)
            if len(float_conversions) > 0:
                # No text header — first column is the timestamp
                dataframe = pd.read_csv(csv_path, header=None)
                timestamp_column = 0
                has_text_header = False
            else:
                dataframe = df_probe
                timestamp_column = next(
                    (column for column in dataframe.columns if TIMESTAMP_COLUMN_SUBSTRING in str(column).lower()),
                    dataframe.columns[0],
                )
                has_text_header = True
            first_timestamp = float(dataframe[timestamp_column].iloc[0])
            # Heuristic: timestamps > 1e6 are in milliseconds (e.g. ComputerTimestamp)
            unit_factor = 1000.0 if first_timestamp > 1e6 else 1.0
            cutoff = first_timestamp + duration_in_seconds * unit_factor
            dataframe = dataframe[dataframe[timestamp_column] <= cutoff]
            dataframe.to_csv(csv_path, index=False, header=has_text_header)

    @classmethod
    def has_multiple_event_ttls(cls, folder_path: str) -> list[bool]:
        """
        Check whether any NPM event files in the folder contain multiple TTL types.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing NPM CSV files.

        Returns
        -------
        multiple_event_ttls : list of bool
            One entry per NPM data file. ``True`` if the corresponding event
            file encodes more than one unique TTL state value, ``False`` otherwise.
        """
        multiple_event_ttls = []
        for path in cls._list_npm_files(folder_path):
            flag, df, _ = cls._classify_npm_file(path)

            if flag == "event_np":
                type_val = np.array(df.iloc[:, 1])
                type_val_unique = np.unique(type_val)
                if len(type_val_unique) > 1:
                    multiple_event_ttls.append(True)
                else:
                    multiple_event_ttls.append(False)
            else:
                multiple_event_ttls.append(False)

        return multiple_event_ttls

    @classmethod
    def timestamp_column_options(cls, folder_path: str) -> list[str]:
        """
        List the timestamp columns the session's data files offer.

        Newer NPM exports can carry more than one timestamp column (e.g. both
        ``Timestamp`` and ``ComputerTimestamp``), which are on different clocks,
        so the user has to say which one to use.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing NPM CSV files.

        Returns
        -------
        list of str
            Distinct timestamp-like column names across the session's data
            files, in the order encountered. Empty for sessions whose files
            carry no text header.
        """
        column_options: list[str] = []
        for path in cls._list_npm_files(folder_path):
            flag, df, _ = cls._classify_npm_file(path)
            if flag == "event_np":
                continue
            for name in df.columns:
                if TIMESTAMP_COLUMN_SUBSTRING in str(name).lower() and name not in column_options:
                    column_options.append(str(name))

        return column_options
