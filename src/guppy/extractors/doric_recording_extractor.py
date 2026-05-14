import glob
import logging
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from guppy.extractors import BaseRecordingExtractor
from guppy.extractors.detect_acquisition_formats import _is_event_csv
from guppy.utils._hdf5_io import write_hdf5

logger = logging.getLogger(__name__)


class DoricRecordingExtractor(BaseRecordingExtractor):
    """
    Extractor for fiber photometry data from Doric Lenses acquisition systems.

    Supports two file formats:

    * **doric_csv** — Doric-exported CSV files with a two-row header.
    * **doric_doric** — Doric HDF5 files (``.doric``), V1 and V6 layouts.

    Parameters
    ----------
    folder_path : str
        Path to the session folder containing the Doric data file.
    event_name_to_event_type : dict
        Mapping from channel/event name (str) to its role in the pipeline
        (e.g. ``"control_DMS"``, ``"signal_DMS"``, ``"ttl_DMS"``).
    """

    # TODO: consolidate duplicate flag logic between the `discover_events_and_flags` and the `check_doric` method.

    @classmethod
    def discover_events_and_flags(cls, folder_path: str) -> tuple[list[str], list[str]]:
        """
        Discover available events and file format flags from Doric files.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing Doric files

        Returns
        -------
        events : list
            List of discovered event names
        flags : list
            List of format flags (e.g., 'doric_csv', 'doric_doric')
        """
        logger.debug("Discovering Doric events from file headers")
        path = sorted(glob.glob(os.path.join(folder_path, "*.csv"))) + sorted(
            glob.glob(os.path.join(folder_path, "*.doric"))
        )
        # Exclude event CSV files (single 'timestamps' column) — those belong to CsvRecordingExtractor
        path = [p for p in path if not (p.endswith(".csv") and _is_event_csv(p))]
        path = sorted(list(set(path)))
        flag = "None"
        event_from_filename = []
        flag_arr = []

        for i in range(len(path)):
            ext = os.path.basename(path[i]).split(".")[-1]
            if ext == "doric":
                key_names = cls._read_doric_file(path[i])
                event_from_filename.extend(key_names)
                flag = "doric_doric"
            else:
                df = pd.read_csv(path[i], header=None, nrows=2, index_col=False, dtype=str)
                df = df.dropna(axis=1, how="all")
                df_arr = np.array(df).flatten()
                check_all_str = []
                for element in df_arr:
                    try:
                        float(element)
                    except:
                        check_all_str.append(i)
                if len(check_all_str) != len(df_arr):
                    raise ValueError(
                        f"CSV file '{path[i]}' appears to be a standard .csv (numeric values in header rows). "
                        "DoricRecordingExtractor only supports Doric .csv files; use the standard CSV extractor instead."
                    )
                df = pd.read_csv(path[i], header=1, index_col=False, nrows=10)
                # Drop trailing all-NaN columns (e.g. "Unnamed: 7" from Doric CSVs with a
                # trailing comma on each line). They'd otherwise appear as selectable events
                # in Step 2, then vanish via the read path's own dropna, yielding a cryptic KeyError.
                df = df.dropna(axis=1, how="all")
                df = df.drop(["Time(s)"], axis=1)
                event_from_filename.extend(list(df.columns))
                flag = "doric_csv"
                logger.info(flag)

        logger.info("Doric event discovery complete.")
        return event_from_filename, flag_arr

    def __init__(self, folder_path: str, event_name_to_event_type: dict[str, str]) -> None:
        self.folder_path = folder_path
        self._event_name_to_event_type = event_name_to_event_type

    @staticmethod
    def _read_doric_file(filepath: str) -> list[str]:
        """Static helper to read Doric file headers for event discovery."""
        with h5py.File(filepath, "r") as f:
            if "Traces" in list(f.keys()):
                keys = DoricRecordingExtractor._access_keys_doricV1(f)
            elif list(f.keys()) == ["Configurations", "DataAcquisition"]:
                keys = DoricRecordingExtractor._access_keys_doricV6(f)

        return keys

    @staticmethod
    def _access_keys_doricV6(doric_file: h5py.File) -> list[str]:
        data = [doric_file["DataAcquisition"]]
        res = []
        while len(data) != 0:
            members = len(data)
            while members != 0:
                members -= 1
                data, last_element = DoricRecordingExtractor._separate_last_element(data)
                if isinstance(last_element, h5py.Dataset) and not last_element.name.endswith("/Time"):
                    res.append(last_element.name)
                elif isinstance(last_element, h5py.Group):
                    data.extend(reversed([last_element[k] for k in last_element.keys()]))

        keys = []
        for element in res:
            sep_values = element.split("/")
            if sep_values[-1] == "Values":
                keys.append(f"{sep_values[-3]}/{sep_values[-2]}")
            else:
                keys.append(f"{sep_values[-2]}/{sep_values[-1]}")

        return keys

    @staticmethod
    def _access_keys_doricV1(doric_file: h5py.File) -> list[str]:
        keys = list(doric_file["Traces"]["Console"].keys())
        keys.remove("Time(s)")

        return keys

    @staticmethod
    def _separate_last_element(arr: list) -> tuple[list, object]:
        l = arr[-1]
        return arr[:-1], l

    @staticmethod
    def _validate_signal_control_data(event: str, data: np.ndarray, event_type: str) -> None:
        """Raise ValueError if ``data`` is unusable as a photometry signal/control trace.

        Rejects channels that are all-NaN, mostly-NaN (>=50%), or zero-variance —
        the three failure modes seen in real Doric exports (unused demodulation
        channels fill with NaN; AOut / LED-drive channels are constant).
        """
        if data.size == 0:
            raise ValueError(f"Doric channel {event!r} is empty and cannot be used as a {event_type} channel.")
        finite_mask = np.isfinite(data)
        nonfinite_count = int((~finite_mask).sum())
        if nonfinite_count > 0:
            # Even one NaN/inf breaks the downstream np.polyfit in z_score.controlFit (SVD error).
            # This is common in Doric 'AIn-X - Dem (AOut-Y)' channels where AOut-Y was not
            # driven (all-NaN demod output) or during demodulator warmup at recording start.
            raise ValueError(
                f"Doric channel {event!r} contains {nonfinite_count} non-finite value(s) "
                f"(NaN/inf) out of {data.size} and cannot be used as a {event_type} channel — "
                "downstream control-fit (np.polyfit) cannot handle NaN. This typically happens "
                "with demodulated channels like 'AIn-X - Dem (AOut-Y)' when excitation "
                "AOut-Y was not driven for this input, or during demodulator warmup at "
                "recording start. Re-run Step 2 and pick a channel with continuous finite data."
            )
        if float(np.std(data)) == 0.0:
            raise ValueError(
                f"Doric channel {event!r} is constant (std=0) and cannot be used as a "
                f"{event_type} channel. This is typical of analog output ('AOut') / LED "
                "excitation channels, which do not contain fluorescence data. Pick a "
                "demodulated or raw photometry channel instead."
            )

    def count_samples(self, *, event: str) -> int:
        """Return the number of samples for ``event`` using only metadata reads."""
        flag = self._check_doric()
        if flag == "doric_csv":
            csv_paths = glob.glob(os.path.join(self.folder_path, "*.csv"))
            if not csv_paths:
                return 0
            with open(csv_paths[0], "rb") as file:
                total_lines = sum(1 for _ in file)
            return max(0, total_lines - 2)
        if flag == "doric_doric":
            doric_paths = glob.glob(os.path.join(self.folder_path, "*.doric"))
            if not doric_paths:
                return 0
            with h5py.File(doric_paths[0], "r") as f:
                if "Traces" in list(f.keys()):
                    console = f["Traces"]["Console"]
                    if event in console:
                        return int(console[event][event].shape[0])
                    return 0
                if list(f.keys()) == ["Configurations", "DataAcquisition"]:
                    data = [f["DataAcquisition"]]
                    res = []
                    while len(data) != 0:
                        members = len(data)
                        while members != 0:
                            members -= 1
                            data, last_element = self._separate_last_element(data)
                            if isinstance(last_element, h5py.Dataset) and not last_element.name.endswith("/Time"):
                                res.append(last_element.name)
                            elif isinstance(last_element, h5py.Group):
                                data.extend(reversed([last_element[k] for k in last_element.keys()]))
                    for element in res:
                        sep_values = element.split("/")
                        if sep_values[-1] == "Values":
                            label = f"{sep_values[-3]}/{sep_values[-2]}"
                        else:
                            label = f"{sep_values[-2]}/{sep_values[-1]}"
                        if label == event:
                            return int(f[element].shape[0])
                    return 0
        return 0

    def _check_doric(self) -> str | int:
        logger.debug("Checking if doric file exists")
        path = glob.glob(os.path.join(self.folder_path, "*.csv")) + glob.glob(os.path.join(self.folder_path, "*.doric"))

        flag_arr = []
        for i in range(len(path)):
            ext = os.path.basename(path[i]).split(".")[-1]
            if ext == "csv":
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    try:
                        df = pd.read_csv(path[i], index_col=False, dtype=float)
                    except:  # TODO: fix this bare try-except
                        df = pd.read_csv(path[i], header=1, index_col=False, nrows=10)
                        flag = "doric_csv"
                        flag_arr.append(flag)
            elif ext == "doric":
                flag = "doric_doric"
                flag_arr.append(flag)
            else:
                pass

        if len(flag_arr) > 1:
            doric_paths = sorted(
                glob.glob(os.path.join(self.folder_path, "*.csv"))
                + glob.glob(os.path.join(self.folder_path, "*.doric"))
            )
            message = (
                f"Multiple Doric data files found in '{self.folder_path}': {doric_paths}. "
                "Each session folder must contain exactly one Doric file."
            )
            logger.error(message)
            raise ValueError(message)
        if len(flag_arr) == 0:
            logger.error("Doric file not found.")
            return 0
        logger.info("Doric file found.")
        return flag_arr[0]

    def _read_doric_csv(self, events: list[str]) -> list[dict[str, object]]:
        path = glob.glob(os.path.join(self.folder_path, "*.csv"))
        if len(path) > 1:
            message = (
                f"Multiple Doric .csv files found in '{self.folder_path}': {sorted(path)}. "
                "Each Doric session folder must contain exactly one .csv file."
            )
            logger.error(message)
            raise ValueError(message)

        df = pd.read_csv(path[0], header=1, index_col=False)
        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=0, how="any")
        df["Time(s)"] = df["Time(s)"] - df["Time(s)"].to_numpy()[0]

        output_dicts = []
        for event in events:
            if event not in df.columns:
                available = sorted(c for c in df.columns if c != "Time(s)")
                raise ValueError(
                    f"Doric channel {event!r} not found in Doric CSV file. "
                    f"Available channels: {available}. Empty columns (trailing commas, "
                    "'Unnamed: N') are filtered during discovery."
                )
            event_type = self._event_name_to_event_type[event]
            if "control" in event_type or "signal" in event_type:
                timestamps = np.array(df["Time(s)"])
                sampling_rate = np.array([1 / (timestamps[-1] - timestamps[-2])])
                data = np.array(df[event])
                self._validate_signal_control_data(event, data, event_type)
                storename = event
                event_dict = {
                    "storename": storename,
                    "sampling_rate": sampling_rate,
                    "timestamps": timestamps,
                    "data": data,
                }
                output_dicts.append(event_dict)
            else:
                ttl = df[event]
                indices = np.where(ttl <= 0)[0]
                diff_indices = np.where(np.diff(indices) > 1)[0]
                timestamps = df["Time(s)"][indices[diff_indices] + 1].to_numpy()
                storename = event
                event_dict = {"storename": storename, "timestamps": timestamps}
                output_dicts.append(event_dict)

        return output_dicts

    def _read_doric_doric(self, events: list[str]) -> list[dict[str, object]]:
        path = glob.glob(os.path.join(self.folder_path, "*.doric"))
        if len(path) > 1:
            message = (
                f"Multiple Doric .doric files found in '{self.folder_path}': {sorted(path)}. "
                "Each Doric session folder must contain exactly one .doric file."
            )
            logger.error(message)
            raise ValueError(message)
        with h5py.File(path[0], "r") as f:
            if "Traces" in list(f.keys()):
                output_dicts = self._access_data_doricV1(f, events)
            elif list(f.keys()) == ["Configurations", "DataAcquisition"]:
                output_dicts = self._access_data_doricV6(f, events)
        return output_dicts

    def _access_data_doricV6(self, doric_file: h5py.File, events: list[str]) -> list[dict[str, object]]:
        data = [doric_file["DataAcquisition"]]
        res = []
        while len(data) != 0:
            members = len(data)
            while members != 0:
                members -= 1
                data, last_element = self._separate_last_element(data)
                if isinstance(last_element, h5py.Dataset) and not last_element.name.endswith("/Time"):
                    res.append(last_element.name)
                elif isinstance(last_element, h5py.Group):
                    data.extend(reversed([last_element[k] for k in last_element.keys()]))

        decide_path = []
        for element in res:
            sep_values = element.split("/")
            if sep_values[-1] == "Values":
                if f"{sep_values[-3]}/{sep_values[-2]}" in events:
                    decide_path.append(element)
            else:
                if f"{sep_values[-2]}/{sep_values[-1]}" in events:
                    decide_path.append(element)

        output_dicts = []
        for event in events:
            event_type = self._event_name_to_event_type[event]
            if "control" in event_type or "signal" in event_type:
                regex = re.compile("(.*?)" + str(event) + "(.*?)")
                idx = [i for i in range(len(decide_path)) if regex.match(decide_path[i])]
                if len(idx) > 1:
                    matched_paths = [decide_path[i] for i in idx]
                    message = (
                        f"Doric channel {event!r} matches multiple internal HDF5 paths "
                        f"({matched_paths}); expected exactly one. The Doric file may have a malformed structure."
                    )
                    logger.error(message)
                    raise ValueError(message)
                if len(idx) == 0:
                    available = sorted(
                        {
                            p.rsplit("/", 1)[-1] if p.rsplit("/", 1)[-1] != "Values" else p.rsplit("/", 2)[-2]
                            for p in res
                        }
                    )
                    raise ValueError(
                        f"Doric channel {event!r} not found in Doric V6 file. " f"Available channels: {available}."
                    )
                idx = idx[0]
                data = np.array(doric_file[decide_path[idx]])
                timestamps = np.array(doric_file[decide_path[idx].rsplit("/", 1)[0] + "/Time"])
                self._validate_signal_control_data(event, data, event_type)
                sampling_rate = np.array([1 / (timestamps[-1] - timestamps[-2])])
                storename = event
                event_dict = {
                    "storename": storename,
                    "sampling_rate": sampling_rate,
                    "timestamps": timestamps,
                    "data": data,
                }
                output_dicts.append(event_dict)
            else:
                regex = re.compile("(.*?)" + event + "$")
                idx = [i for i in range(len(decide_path)) if regex.match(decide_path[i])]
                if len(idx) > 1:
                    matched_paths = [decide_path[i] for i in idx]
                    message = (
                        f"Doric channel {event!r} matches multiple internal HDF5 paths "
                        f"({matched_paths}); expected exactly one. The Doric file may have a malformed structure."
                    )
                    logger.error(message)
                    raise ValueError(message)
                if len(idx) == 0:
                    available = sorted(
                        {
                            p.rsplit("/", 1)[-1] if p.rsplit("/", 1)[-1] != "Values" else p.rsplit("/", 2)[-2]
                            for p in res
                        }
                    )
                    raise ValueError(
                        f"Doric TTL channel {event!r} not found in Doric V6 file. " f"Available channels: {available}."
                    )
                idx = idx[0]
                ttl = np.array(doric_file[decide_path[idx]])
                timestamps = np.array(doric_file[decide_path[idx].rsplit("/", 1)[0] + "/Time"])
                indices = np.where(ttl <= 0)[0]
                diff_indices = np.where(np.diff(indices) > 1)[0]
                timestamps = timestamps[indices[diff_indices] + 1]
                storename = event
                event_dict = {"storename": storename, "timestamps": timestamps}
                output_dicts.append(event_dict)

        return output_dicts

    def _access_data_doricV1(self, doric_file: h5py.File, events: list[str]) -> list[dict[str, object]]:
        console = doric_file["Traces"]["Console"]
        output_dicts = []
        for event in events:
            if event not in console:
                available = sorted(k for k in console.keys() if k != "Time(s)")
                raise ValueError(
                    f"Doric channel {event!r} not found in Doric V1 file. " f"Available channels: {available}."
                )
            event_type = self._event_name_to_event_type[event]
            if "control" in event_type or "signal" in event_type:
                timestamps = np.array(console["Time(s)"]["Console_time(s)"])
                data = np.array(console[event][event])
                self._validate_signal_control_data(event, data, event_type)
                sampling_rate = np.array([1 / (timestamps[-1] - timestamps[-2])])
                storename = event
                event_dict = {
                    "storename": storename,
                    "sampling_rate": sampling_rate,
                    "timestamps": timestamps,
                    "data": data,
                }
                output_dicts.append(event_dict)
            else:
                timestamps = np.array(console["Time(s)"]["Console_time(s)"])
                ttl = np.array(console[event][event])
                indices = np.where(ttl <= 0)[0]
                diff_indices = np.where(np.diff(indices) > 1)[0]
                timestamps = timestamps[indices[diff_indices] + 1]
                storename = event
                event_dict = {"storename": storename, "timestamps": timestamps}
                output_dicts.append(event_dict)

        return output_dicts

    def read(self, *, events: list[str], outputPath: str) -> list[dict[str, Any]]:
        """
        Read data from Doric files for the specified events.

        Detects the Doric file format (CSV or HDF5) and dispatches to the
        appropriate reader. Signal/control channels are returned with full
        time-series data; TTL channels are returned as onset timestamps only.

        Parameters
        ----------
        events : list of str
            Channel/event names to read.
        outputPath : str
            Path to the output directory (unused by this extractor; required by
            the base-class interface).

        Returns
        -------
        list of dict
            One dictionary per event. Signal/control dicts contain
            ``storename``, ``sampling_rate``, ``timestamps``, and ``data``;
            TTL dicts contain ``storename`` and ``timestamps``.
        """
        flag = self._check_doric()
        if flag == "doric_csv":
            output_dicts = self._read_doric_csv(events)
        elif flag == "doric_doric":
            output_dicts = self._read_doric_doric(events)
        else:
            message = (
                f"No Doric file (.doric or .csv) found in '{self.folder_path}', or the file format "
                "could not be recognized."
            )
            logger.error(message)
            raise FileNotFoundError(message)

        return output_dicts

    def stub(self, *, folder_path: Path | str, duration_in_seconds: float = 1.0) -> None:
        """
        Create a stubbed copy of the Doric folder truncated to a short duration.

        Copies the source folder to `folder_path`, then rewrites the Doric data
        file so that only the first `duration_in_seconds` of recorded data are
        retained. Supports V1 (.doric HDF5) and CSV (doric_csv) formats. V6
        HDF5 format is not yet supported.

        Parameters
        ----------
        folder_path : str or Path
            Destination directory for the stubbed folder. Created if it does
            not exist; overwritten if it already exists.
        duration_in_seconds : float, optional
            Approximate duration of data to retain in seconds. Default is 1.0.
        """
        folder_path = Path(folder_path)
        if folder_path.exists():
            shutil.rmtree(folder_path)
        shutil.copytree(self.folder_path, folder_path)

        flag = self._check_doric()
        if flag == "doric_doric":
            self._stub_doric_hdf5(folder_path=folder_path, duration_in_seconds=duration_in_seconds)
        elif flag == "doric_csv":
            self._stub_doric_csv(folder_path=folder_path, duration_in_seconds=duration_in_seconds)

    def _stub_doric_hdf5(self, *, folder_path: Path | str, duration_in_seconds: float) -> None:
        doric_paths = glob.glob(os.path.join(folder_path, "*.doric"))
        doric_path = doric_paths[0]

        with h5py.File(doric_path, "r") as source_file:
            if "Traces" in list(source_file.keys()):
                temporary_path = self._stub_doric_hdf5_v1(
                    source_file=source_file, doric_path=doric_path, duration_in_seconds=duration_in_seconds
                )
            else:
                temporary_path = self._stub_doric_hdf5_v6(
                    source_file=source_file, doric_path=doric_path, duration_in_seconds=duration_in_seconds
                )

        # Replace after closing source_file so Windows does not raise PermissionError on open files.
        os.replace(temporary_path, doric_path)

    def _stub_doric_hdf5_v1(self, *, source_file: h5py.File, doric_path: str, duration_in_seconds: float) -> str:
        timestamps = np.array(source_file["Traces"]["Console"]["Time(s)"]["Console_time(s)"])
        cutoff_timestamp = timestamps[0] + duration_in_seconds
        cutoff_index = int(np.searchsorted(timestamps, cutoff_timestamp, side="right"))

        # Only keep channels needed by this extractor instance, not all channels in the file.
        # This avoids copying irrelevant channels that inflate stub file size.
        channel_keys = list(self._event_name_to_event_type.keys())

        channel_data = {}
        for key in channel_keys:
            channel_data[key] = np.array(source_file["Traces"]["Console"][key][key])

        temporary_path = doric_path + ".tmp"
        with h5py.File(temporary_path, "w") as destination_file:
            console = destination_file.require_group("Traces/Console")
            time_group = console.require_group("Time(s)")
            time_group.create_dataset("Console_time(s)", data=timestamps[:cutoff_index], compression="gzip")
            for key in channel_keys:
                channel_group = console.require_group(key)
                channel_group.create_dataset(key, data=channel_data[key][:cutoff_index], compression="gzip")

        return temporary_path

    def _stub_doric_hdf5_v6(self, *, source_file: h5py.File, doric_path: str, duration_in_seconds: float) -> str:
        temporary_path = doric_path + ".tmp"
        with h5py.File(temporary_path, "w") as destination_file:
            if "Configurations" in source_file:
                source_file.copy("Configurations", destination_file)
            self._copy_group_truncated(
                source_group=source_file["DataAcquisition"],
                destination_group=destination_file.require_group("DataAcquisition"),
                duration_in_seconds=duration_in_seconds,
            )
        return temporary_path

    def _copy_group_truncated(
        self, *, source_group: h5py.Group, destination_group: h5py.Group, duration_in_seconds: float
    ) -> None:
        if "Time" in source_group:
            time_data = source_group["Time"][:]
            cutoff_index = int(np.searchsorted(time_data, time_data[0] + duration_in_seconds, side="right"))
            for key in source_group:
                destination_group.create_dataset(key, data=source_group[key][:cutoff_index])
        else:
            for key in source_group:
                item = source_group[key]
                if isinstance(item, h5py.Group):
                    self._copy_group_truncated(
                        source_group=item,
                        destination_group=destination_group.require_group(key),
                        duration_in_seconds=duration_in_seconds,
                    )
                else:
                    destination_group.create_dataset(key, data=item[:])

    def _stub_doric_csv(self, *, folder_path: Path | str, duration_in_seconds: float) -> None:
        csv_paths = glob.glob(os.path.join(folder_path, "*.csv"))
        csv_path = csv_paths[0]

        # Row 0 is the channel descriptor row; row 1 is the column name row (header=1 skips both)
        header_rows = pd.read_csv(csv_path, header=None, nrows=2, index_col=False, dtype=str)
        dataframe = pd.read_csv(csv_path, header=1, index_col=False)
        dataframe = dataframe.dropna(axis=1, how="all")

        cutoff_timestamp = dataframe["Time(s)"].iloc[0] + duration_in_seconds
        dataframe = dataframe[dataframe["Time(s)"] <= cutoff_timestamp]

        with open(csv_path, "w") as file:
            header_rows.to_csv(file, index=False, header=False)
            dataframe.to_csv(file, index=False, header=False)

    def save(self, *, output_dicts: list[dict[str, Any]], outputPath: str) -> None:
        """
        Save extracted data dictionaries to HDF5 files.

        Parameters
        ----------
        output_dicts : list of dict
            Data dictionaries as returned by :meth:`read`.
        outputPath : str
            Path to the output directory where HDF5 files are written.
        """
        for event_dict in output_dicts:
            storename = event_dict["storename"]
            write_hdf5(data=event_dict["timestamps"], storename=storename, output_path=outputPath, key="timestamps")

            if "sampling_rate" in event_dict:
                write_hdf5(
                    data=event_dict["sampling_rate"], storename=storename, output_path=outputPath, key="sampling_rate"
                )
            if "data" in event_dict:
                write_hdf5(data=event_dict["data"], storename=storename, output_path=outputPath, key="data")
