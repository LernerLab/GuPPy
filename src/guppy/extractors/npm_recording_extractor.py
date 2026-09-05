import logging
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


class NpmRecordingExtractor(CsvRecordingExtractor):
    """
    Extractor for fiber photometry data from Neurophotometrics (NPM) systems.

    NPM files store interleaved channels (and, optionally, a multi-type event
    column) in multi-column CSVs. This extractor demultiplexes those raw files
    into per-channel and per-event streams **entirely in memory** via
    :meth:`decompose`; ``read`` and ``count_samples`` are thin wrappers over it.
    Nothing is written to the source folder — only the final HDF5 outputs are
    written, by the inherited :meth:`CsvRecordingExtractor.save`.

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

        streams, flags = cls._decompose_streams(
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
            Sorted paths of the raw NPM files. The index of a path in this list
            is the ``file{i}_`` prefix its channels are named with, and the
            index its ``npm_split_events`` entry is read from.
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
    ) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
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
            (``file{i}_chev{j}`` / ``chod{j}`` / ``chpr{j}``) carry
            ``timestamps``, ``data``, and ``sampling_rate``; event streams
            (``event{value}`` / ``event0``) carry only ``timestamps``.
        flags : list of str
            One format flag per raw source file processed.
        """
        logger.debug("If it exists, importing NPM file based on the structure of file")
        divisor = cls._time_unit_divisor(npm_time_unit)
        path = cls._list_npm_files(folder_path)

        streams: dict[str, dict[str, np.ndarray]] = {}
        # Track derived stream names in creation order so the cross-file channel
        # pairing (chod/chpr borrow their paired chev) is deterministic.
        chev_names: list[str] = []
        chod_names: list[str] = []
        chpr_names: list[str] = []
        flags: list[str] = []
        for i in range(len(path)):
            # TODO: validate npm_split_events length
            if npm_split_events is None:
                split_events = False
            else:
                split_events = npm_split_events[i]

            flag, df, _ = cls._classify_npm_file(path[i])

            flags.append(flag)
            logger.info(flag)
            if flag == "data_np":
                file_prefix = f"file{str(i)}_"
                df, indices_dict, _ = cls.decide_indices(file_prefix, df, flag, num_ch)
                keys = list(indices_dict.keys())
                for k in range(len(keys)):
                    for j in range(df.shape[1]):
                        if j == 0:
                            timestamps = df.iloc[:, j][indices_dict[keys[k]]]
                        else:
                            name = keys[k] + str(j)
                            streams[name] = {
                                "timestamps": np.asarray(timestamps, dtype=float),
                                "data": np.asarray(df.iloc[:, j][indices_dict[keys[k]]], dtype=float),
                            }
                            cls._register_channel_name(name, keys[k], chev_names, chod_names, chpr_names)

            elif flag == "event_np":
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
            else:
                file_prefix = f"file{str(i)}_"
                df = cls._update_df_with_timestamp_columns(df, timestamp_column_name=npm_timestamp_column_name)
                df, indices_dict, _ = cls.decide_indices(file_prefix, df, flag)
                keys = list(indices_dict.keys())
                for k in range(len(keys)):
                    for j in range(df.shape[1]):
                        if j == 0:
                            timestamps = df.iloc[:, j][indices_dict[keys[k]]]
                        else:
                            name = keys[k] + str(j)
                            streams[name] = {
                                "timestamps": np.asarray(timestamps, dtype=float),
                                "data": np.asarray(df.iloc[:, j][indices_dict[keys[k]]], dtype=float),
                            }
                            cls._register_channel_name(name, keys[k], chev_names, chod_names, chpr_names)

        # Convert every stream to seconds with the session's single timestamp unit, then
        # compute sampling rates. Timestamps keep the acquisition's own clock.
        for stream in streams.values():
            stream["timestamps"] = stream["timestamps"] / divisor

        channel_group_lengths = [len(names) for names in (chev_names, chod_names, chpr_names) if len(names) > 0]
        if len(set(channel_group_lengths)) > 1:
            channel_group_counts = {
                "chev": len(chev_names),
                "chod": len(chod_names),
                "chpr": len(chpr_names),
            }
            message = (
                "Number of channel files must match across channel groups (chev/chod/chpr). "
                f"Found per-channel-group counts: {channel_group_counts}."
            )
            logger.error(message)
            raise ValueError(message)

        # The paired chod/chpr channels borrow chev's timestamps and rate.
        for j in range(len(chev_names)):
            chev_stream = streams[chev_names[j]]
            chev_timestamps = chev_stream["timestamps"]
            sampling_rate = chev_timestamps.shape[0] / (chev_timestamps[-1] - chev_timestamps[0])
            chev_stream["sampling_rate"] = np.array([sampling_rate])

            if j < len(chod_names):
                chod_stream = streams[chod_names[j]]
                chod_stream["timestamps"] = chev_timestamps
                chod_stream["sampling_rate"] = np.array([sampling_rate])
            if j < len(chpr_names):
                chpr_stream = streams[chpr_names[j]]
                chpr_stream["timestamps"] = chev_timestamps
                chpr_stream["sampling_rate"] = np.array([sampling_rate])

        logger.info("Importing of NPM file is done.")
        return streams, flags

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

    @staticmethod
    def _register_channel_name(
        name: str, channel_key: str, chev_names: list[str], chod_names: list[str], chpr_names: list[str]
    ) -> None:
        """Append ``name`` to the creation-order list for its channel group."""
        if "chev" in channel_key:
            chev_names.append(name)
        elif "chod" in channel_key:
            chod_names.append(name)
        elif "chpr" in channel_key:
            chpr_names.append(name)

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
            streams, _ = self._decompose_streams(
                folder_path=self.folder_path,
                num_ch=self.num_ch,
                npm_timestamp_column_name=self.npm_timestamp_column_name,
                npm_time_unit=self.npm_time_unit,
                npm_split_events=self.npm_split_events,
            )
            self._decomposed = streams
        return self._decomposed

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
                    (column for column in dataframe.columns if "timestamp" in column.lower()),
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

    # function to decide indices of interleaved channels
    # in neurophotometrics data
    @classmethod
    def decide_indices(
        cls, file_prefix: str, df: pd.DataFrame, flag: str, num_ch: int = 2
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray], int]:
        """
        Determine the row indices belonging to each interleaved channel in NPM data.

        For older NPM layouts (``data_np``), rows are assumed to cycle through
        channels in order. For newer layouts (``data_np_v2``), the ``Flags`` or
        ``LedState`` column is used to assign rows to channels.

        Parameters
        ----------
        file_prefix : str
            Filename prefix used to build per-channel keys (e.g. ``"file0_"``).
        df : pd.DataFrame
            NPM data DataFrame, with or without a text header.
        flag : str
            Layout flag as returned by :meth:`discover_events_and_flags`.
        num_ch : int, optional
            Number of interleaved channels expected. Default is 2.

        Returns
        -------
        df : pd.DataFrame
            The input DataFrame with flag columns (``Flags`` / ``LedState``)
            dropped for ``data_np_v2`` layouts.
        indices_dict : dict
            Mapping from channel key (e.g. ``"file0_chev"``) to a NumPy array
            of row indices belonging to that channel.
        num_ch : int
            Actual number of channels detected.
        """
        channel_keys = [file_prefix + "chev", file_prefix + "chod", file_prefix + "chpr"]
        if len(channel_keys) < num_ch:
            message = (
                f"Number of channels in the Input Parameters GUI is set to {num_ch}, which exceeds the "
                "maximum of 3 channels supported for NPM files. Set 'Number of channels' to 3 or fewer "
                "in the Input Parameters GUI."
            )
            logger.error(message)
            raise ValueError(message)
        if flag == "data_np":
            indices_dict = dict()
            for i in range(num_ch):
                indices_dict[channel_keys[i]] = np.arange(i, df.shape[0], num_ch)

        else:
            columns = np.array(list(df.columns))
            # Detection matches these columns case-insensitively, so resolve the
            # actual (possibly mixed-case) column names before indexing.
            column_by_name = cls._column_by_lowercase_name(df)
            if "flags" in column_by_name:
                columns_to_drop = [column_by_name["framecounter"], column_by_name["flags"]]
                state = np.array(df[column_by_name["flags"]])
            elif "ledstate" in column_by_name:
                columns_to_drop = [column_by_name["framecounter"], column_by_name["ledstate"]]
                state = np.array(df[column_by_name["ledstate"]])
            else:
                message = (
                    "File type indicates Neurophotometrics newer version data but the columns do not "
                    f"contain a 'Flags' or 'LedState' column. Found columns: {list(columns)}."
                )
                logger.error(message)
                raise ValueError(message)

            num_ch, unique_states = cls.check_channels(state)
            indices_dict = dict()
            for i in range(num_ch):
                first_occurrence = np.where(state == unique_states[i])[0]
                indices_dict[channel_keys[i]] = np.arange(first_occurrence[0], df.shape[0], num_ch)

            df = df.drop(columns_to_drop, axis=1)

        return df, indices_dict, num_ch

    # check flag consistency in neurophotometrics data
    @classmethod
    def check_channels(cls, state: np.ndarray) -> tuple[int, np.ndarray]:
        """
        Validate and count unique channel states in NPM ``Flags``/``LedState`` data.

        Parameters
        ----------
        state : array-like
            Integer channel-state values (from the ``Flags`` or ``LedState``
            column). Only rows 2–11 are examined to skip potential startup
            artefacts.

        Returns
        -------
        num_ch : int
            Number of unique channel states found.
        unique_state : np.ndarray
            Sorted array of unique state values.
        """
        state = state.astype(int)
        unique_state = np.unique(state[2:12])
        if unique_state.shape[0] > 3:
            message = (
                f"NPM file contains {unique_state.shape[0]} unique channel states ({unique_state.tolist()}), "
                "but only 1-3 channels are supported."
            )
            logger.error(message)
            raise ValueError(message)

        return unique_state.shape[0], unique_state

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
                if "timestamp" in str(name).lower() and name not in column_options:
                    column_options.append(str(name))

        return column_options

    @staticmethod
    def _update_df_with_timestamp_columns(df: pd.DataFrame, timestamp_column_name: str | None) -> pd.DataFrame:
        """Reduce multiple timestamp columns to a single ``Timestamp`` column.

        Files offering only one timestamp column are returned unchanged, so a
        session-wide ``timestamp_column_name`` may name a column that some of
        the session's files do not have.

        Parameters
        ----------
        df : pd.DataFrame
            NPM data file contents, with a text header.
        timestamp_column_name : str or None
            Timestamp column to keep; ``None`` keeps the first one.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the chosen column inserted as ``Timestamp`` and
            the original timestamp columns dropped.
        """
        timestamp_column_names = [name for name in df.columns if "timestamp" in name.lower()]
        if len(timestamp_column_names) <= 1:
            return df

        timestamp_column_name = (
            timestamp_column_name if timestamp_column_name is not None else timestamp_column_names[0]
        )
        if timestamp_column_name not in timestamp_column_names:
            raise ValueError(
                f"Provided timestamp_column_name '{timestamp_column_name}' not found in columns {timestamp_column_names}."
            )
        df.insert(1, "Timestamp", df[timestamp_column_name])
        df = df.drop(timestamp_column_names, axis=1)
        return df
