import glob
import logging
import os
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
    npm_timestamp_column_names : list of str, optional
        Per-file timestamp column names (for files with multiple timestamp
        columns). ``None`` lets the extractor pick the first timestamp column.
    npm_time_units : list of str, optional
        Per-file timestamp units (``"seconds"``, ``"milliseconds"``, or
        ``"microseconds"``). ``None`` defaults to seconds per file.
    npm_split_events : list of bool, optional
        Per-file flag controlling whether a multi-type event column is split
        into one event stream per unique value. ``None`` defaults to no split.
    """

    def __init__(
        self,
        folder_path: str,
        *,
        num_ch: int = 2,
        npm_timestamp_column_names: list[str] | None = None,
        npm_time_units: list[str] | None = None,
        npm_split_events: list[bool] | None = None,
    ) -> None:
        self.folder_path = folder_path
        self.num_ch = num_ch
        self.npm_timestamp_column_names = npm_timestamp_column_names
        self.npm_time_units = npm_time_units
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
        npm_timestamp_column_names = None
        npm_time_units = None
        npm_split_events = None
        if isinstance(inputParameters, dict):
            npm_timestamp_column_names = inputParameters.get("npm_timestamp_column_names")
            npm_time_units = inputParameters.get("npm_time_units")
            # TODO: come up with a better name for npm_split_events that can be appropriately pluralized for a list
            npm_split_events = inputParameters.get("npm_split_events")

        streams, flags = cls._decompose_streams(
            folder_path=folder_path,
            num_ch=num_ch,
            npm_timestamp_column_names=npm_timestamp_column_names,
            npm_time_units=npm_time_units,
            npm_split_events=npm_split_events,
        )
        return list(streams.keys()), flags

    @classmethod
    def _decompose_streams(
        cls,
        *,
        folder_path: str,
        num_ch: int,
        npm_timestamp_column_names: list[str] | None,
        npm_time_units: list[str] | None,
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
        npm_timestamp_column_names : list of str or None
            Per-file timestamp column names; ``None`` for default selection.
        npm_time_units : list of str or None
            Per-file timestamp units; ``None`` for seconds.
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
        path = sorted(glob.glob(os.path.join(folder_path, "*.csv"))) + sorted(
            glob.glob(os.path.join(folder_path, "*.doric"))
        )
        path_chev = glob.glob(os.path.join(folder_path, "*chev*"))
        path_chod = glob.glob(os.path.join(folder_path, "*chod*"))
        path_chpr = glob.glob(os.path.join(folder_path, "*chpr*"))
        path_event = glob.glob(os.path.join(folder_path, "event*"))
        path_chev_chod_event = path_chev + path_chod + path_event + path_chpr

        path = sorted(list(set(path) - set(path_chev_chod_event)))
        # Exclude single-column timestamp CSVs — those are handled by CsvRecordingExtractor.
        path = [csv_path for csv_path in path if not (csv_path.endswith(".csv") and _is_event_csv(csv_path))]

        streams: dict[str, dict[str, np.ndarray]] = {}
        # Track derived stream names in creation order so the cross-file channel
        # pairing (chod/chpr borrow their paired chev) is deterministic.
        chev_names: list[str] = []
        chod_names: list[str] = []
        chpr_names: list[str] = []
        event_names: list[str] = []
        flags: list[str] = []
        ts_unit = "seconds"
        for i in range(len(path)):
            # TODO: validate npm_timestamp_column_names, npm_time_units, npm_split_events lengths
            if npm_timestamp_column_names is None:
                npm_timestamp_column_name = None
            else:
                npm_timestamp_column_name = npm_timestamp_column_names[i]
            if npm_time_units is None:
                npm_time_unit = "seconds"
            else:
                npm_time_unit = npm_time_units[i]
            if npm_split_events is None:
                split_events = False
            else:
                split_events = npm_split_events[i]

            extension = os.path.basename(path[i]).split(".")[-1]
            if extension == "doric":
                raise ValueError(f"Doric files are not supported by NpmRecordingExtractor; got '{path[i]}'.")
            df = pd.read_csv(path[i], header=None, nrows=2, index_col=False, dtype=str)
            df = df.dropna(axis=1, how="all")
            header_values = np.array(df).flatten()
            check_all_str = []
            for element in header_values:
                try:
                    float(element)
                except:
                    check_all_str.append(i)
            if len(check_all_str) == len(header_values):
                raise ValueError(
                    f"CSV file '{path[i]}' appears to be a Doric .csv (all-string header rows). "
                    "NpmRecordingExtractor only supports NPM .csv files; use the Doric extractor instead."
                )
            df = pd.read_csv(path[i], index_col=False)
            _, numeric_headers = cls._check_header(df)

            # check dataframe structure and read data accordingly
            if len(numeric_headers) > 0:
                columns_isstr = False
                df = pd.read_csv(path[i], header=None)
                columns = np.array(list(df.columns), dtype=str)
            else:
                df = df
                columns_isstr = True
                columns = np.array(list(df.columns), dtype=str)
            # check the structure of dataframe and assign flag to the type of file
            if len(columns) == 1:
                raise ValueError(
                    f"CSV file '{path[i]}' has 1 column (event .csv layout). "
                    "NpmRecordingExtractor only supports NPM .csv files; use the standard CSV extractor for event timestamp files."
                )
            if len(columns) == 3:
                raise ValueError(
                    f"CSV file '{path[i]}' has 3 columns {list(columns)} (data .csv layout). "
                    "NpmRecordingExtractor only supports NPM .csv files; use the standard CSV extractor for 3-column data files."
                )
            if len(columns) == 2:
                flag = "event_or_data_np"
            elif len(columns) > 2:
                flag = "data_np"

            if columns_isstr == True and (
                "flags" in np.char.lower(np.array(columns)) or "ledstate" in np.char.lower(np.array(columns))
            ):
                flag = flag + "_v2"
            else:
                flag = flag

            # used assigned flags to process the files and read the data
            if flag == "event_or_data_np":
                second_column_values = list(df.iloc[:, 1])
                check_float = [True for value in second_column_values if isinstance(value, float)]
                if len(second_column_values) == len(check_float) and columns_isstr == False:
                    flag = "data_np"
                elif columns_isstr == True and ("value" in np.char.lower(np.array(columns))):
                    flag = "event_np"
                else:
                    flag = "event_np"

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
                        event_names.append(name)
                else:
                    timestamps = np.array(df.iloc[:, 0])
                    name = "event" + str(0)
                    streams[name] = {"timestamps": np.asarray(timestamps, dtype=float)}
                    event_names.append(name)
            else:
                file_prefix = f"file{str(i)}_"
                ts_unit = npm_time_unit
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

        # Normalize timestamps relative to the chev reference and compute sampling
        # rates. Only runs when at least one data channel is present.
        if "data_np_v2" in flags or "data_np" in flags:
            if "data_np_v2" in flags:
                if ts_unit == "seconds":
                    divisor = 1
                elif ts_unit == "milliseconds":
                    divisor = 1e3
                else:
                    divisor = 1e6
            else:
                divisor = 1000

            # Events are normalized against the first chev channel's first raw
            # timestamp (captured before chev itself is normalized below).
            if chev_names:
                chev_reference_timestamp = streams[chev_names[0]]["timestamps"][0]
                for name in event_names:
                    streams[name]["timestamps"] = (streams[name]["timestamps"] - chev_reference_timestamp) / divisor

            region_lengths = [len(names) for names in (chev_names, chod_names, chpr_names) if len(names) > 0]
            if len(set(region_lengths)) > 1:
                region_counts = {
                    "chev": len(chev_names),
                    "chod": len(chod_names),
                    "chpr": len(chpr_names),
                }
                message = (
                    "Number of channel files must be the same for all regions. "
                    f"Found per-region counts: {region_counts}."
                )
                logger.error(message)
                raise ValueError(message)

            # Each chev channel is normalized to its own first raw timestamp; the
            # paired chod/chpr channels borrow chev's normalized timestamps and rate.
            for j in range(len(chev_names)):
                chev_stream = streams[chev_names[j]]
                chev_timestamps = (chev_stream["timestamps"] - chev_stream["timestamps"][0]) / divisor
                sampling_rate = chev_timestamps.shape[0] / (chev_timestamps[-1] - chev_timestamps[0])
                chev_stream["timestamps"] = chev_timestamps
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
    def _register_channel_name(
        name: str, channel_key: str, chev_names: list[str], chod_names: list[str], chpr_names: list[str]
    ) -> None:
        """Append ``name`` to the creation-order list for its channel region."""
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
                npm_timestamp_column_names=self.npm_timestamp_column_names,
                npm_time_units=self.npm_time_units,
                npm_split_events=self.npm_split_events,
            )
            self._decomposed = streams
        return self._decomposed

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
            ``storename``, ``timestamps``, ``data``, and ``sampling_rate``;
            event streams produce dicts with keys ``storename`` and
            ``timestamps``.
        """
        streams = self.decompose()
        output_dicts = []
        for event in events:
            output_dicts.append({"storename": event, **streams[event]})
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
        path = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
        path_chev = glob.glob(os.path.join(folder_path, "*chev*"))
        path_chod = glob.glob(os.path.join(folder_path, "*chod*"))
        path_chpr = glob.glob(os.path.join(folder_path, "*chpr*"))
        path_event = glob.glob(os.path.join(folder_path, "event*"))
        path_chev_chod_event = path_chev + path_chod + path_event + path_chpr

        path = sorted(list(set(path) - set(path_chev_chod_event)))
        path = [
            csv_path for csv_path in path if not _is_event_csv(csv_path)
        ]  # Skip event CSVs, which are handled by CsvRecordingExtractor
        multiple_event_ttls = []
        for i in range(len(path)):
            df = pd.read_csv(path[i], index_col=False)
            _, numeric_headers = cls._check_header(df)

            # check dataframe structure and read data accordingly
            if len(numeric_headers) > 0:
                columns_isstr = False
                df = pd.read_csv(path[i], header=None)
                columns = np.array(list(df.columns), dtype=str)
            else:
                columns_isstr = True
                columns = np.array(list(df.columns), dtype=str)
            if len(columns) == 2:
                flag = "event_or_data_np"
            elif len(columns) > 2:
                flag = "data_np"
            else:
                message = (
                    f"CSV file '{path[i]}' has {len(columns)} columns, which is not a recognized NPM layout. "
                    "Expected 2+ columns."
                )
                logger.error(message)
                raise ValueError(message)

            # used assigned flags to process the files and read the data
            if flag == "event_or_data_np":
                second_column_values = list(df.iloc[:, 1])
                check_float = [True for value in second_column_values if isinstance(value, float)]
                if len(second_column_values) == len(check_float) and columns_isstr == False:
                    flag = "data_np"
                elif columns_isstr == True and ("value" in np.char.lower(np.array(columns))):
                    flag = "event_np"
                else:
                    flag = "event_np"

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
            if "flags" in np.char.lower(np.array(columns)):
                columns_to_drop = ["FrameCounter", "Flags"]
                state = np.array(df["Flags"])
            elif "ledstate" in np.char.lower(np.array(columns)):
                columns_to_drop = ["FrameCounter", "LedState"]
                state = np.array(df["LedState"])
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
    def needs_ts_unit(cls, folder_path: str, num_ch: int) -> tuple[list[bool], list[str]]:
        """
        Determine which NPM files require explicit timestamp-unit configuration.

        Files with more than one timestamp-like column (e.g. both
        ``Timestamp`` and ``ComputerTimestamp``) require the user to specify
        which column to use and what units it is in.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing NPM CSV files.
        num_ch : int
            Number of interleaved channels, used when demultiplexing
            ``data_np`` layout files.

        Returns
        -------
        ts_unit_needs : list of bool
            One entry per NPM data file. ``True`` if the file has multiple
            timestamp columns and requires unit disambiguation.
        timestamp_column_names : list of str
            Names of all timestamp-like columns found across the files.
        """
        path = sorted(glob.glob(os.path.join(folder_path, "*.csv"))) + sorted(
            glob.glob(os.path.join(folder_path, "*.doric"))
        )
        path_chev = glob.glob(os.path.join(folder_path, "*chev*"))
        path_chod = glob.glob(os.path.join(folder_path, "*chod*"))
        path_chpr = glob.glob(os.path.join(folder_path, "*chpr*"))
        path_event = glob.glob(os.path.join(folder_path, "event*"))
        # path_sig = glob.glob(os.path.join(filepath, 'sig*')) # TODO: what is this for?
        path_chev_chod_event = path_chev + path_chod + path_event + path_chpr

        path = sorted(list(set(path) - set(path_chev_chod_event)))
        path = [
            csv_path for csv_path in path if not _is_event_csv(csv_path)
        ]  # Skip event CSVs, which are handled by CsvRecordingExtractor
        ts_unit_needs = []
        timestamp_column_names = [""]
        for i in range(len(path)):
            df = pd.read_csv(path[i], index_col=False)
            _, numeric_headers = cls._check_header(df)

            # check dataframe structure and read data accordingly
            if len(numeric_headers) > 0:
                df = pd.read_csv(path[i], header=None)
                columns = np.array(list(df.columns), dtype=str)
                columns_isstr = False
            else:
                columns_isstr = True
                columns = np.array(list(df.columns), dtype=str)
            # check the structure of dataframe and assign flag to the type of file
            if len(columns) == 2:
                flag = "event_or_data_np"
            elif len(columns) > 2:
                flag = "data_np"
            else:
                message = (
                    f"CSV file '{path[i]}' has {len(columns)} columns, which is not a recognized NPM layout. "
                    "Expected 2+ columns."
                )
                logger.error(message)
                raise ValueError(message)

            if columns_isstr == True and (
                "flags" in np.char.lower(np.array(columns)) or "ledstate" in np.char.lower(np.array(columns))
            ):
                flag = flag + "_v2"

            # used assigned flags to process the files and read the data
            if flag == "event_or_data_np":
                second_column_values = list(df.iloc[:, 1])
                check_float = [True for value in second_column_values if isinstance(value, float)]
                if len(second_column_values) == len(check_float) and columns_isstr == False:
                    flag = "data_np"
                elif columns_isstr == True and ("value" in np.char.lower(np.array(columns))):
                    flag = "event_np"
                else:
                    flag = "event_np"

            if flag == "data_np":
                file_prefix = f"file{str(i)}_"
                df, _, _ = cls.decide_indices(file_prefix, df, flag, num_ch)

            if flag == "event_np" or flag == "data_np":
                ts_unit_needs.append(False)
                continue

            column_names = np.array(list(df.columns))
            for name in column_names:
                if "timestamp" in name.lower():
                    timestamp_column_names.append(name)

            if len(timestamp_column_names) > 2:
                ts_unit_needs.append(True)
            else:
                ts_unit_needs.append(False)

        return ts_unit_needs, timestamp_column_names

    @staticmethod
    def _update_df_with_timestamp_columns(df: pd.DataFrame, timestamp_column_name: str | None) -> pd.DataFrame:
        column_names = np.array(list(df.columns))
        timestamp_column_names = [""]
        for name in column_names:
            if "timestamp" in name.lower():
                timestamp_column_names.append(name)
        if len(timestamp_column_names) <= 2:
            return df

        timestamp_column_name = (
            timestamp_column_name if timestamp_column_name is not None else timestamp_column_names[1]
        )
        if timestamp_column_name not in timestamp_column_names:
            raise ValueError(
                f"Provided timestamp_column_name '{timestamp_column_name}' not found in columns {timestamp_column_names[1:]}."
            )
        df.insert(1, "Timestamp", df[timestamp_column_name])
        df = df.drop(timestamp_column_names[1:], axis=1)
        return df
