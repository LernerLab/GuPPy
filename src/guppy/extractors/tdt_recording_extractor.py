import glob
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy import float32, float64, int32, int64, uint16

from guppy.extractors import BaseRecordingExtractor
from guppy.utils._hdf5_io import write_hdf5

logger = logging.getLogger(__name__)


class TdtRecordingExtractor(BaseRecordingExtractor):
    """
    Extractor for fiber photometry data from Tucker-Davis Technologies (TDT) systems.

    Reads binary TDT tank data (``*.tsq`` header + ``*.tev`` data) and extracts
    continuous photometry streams and TTL/epoc events.

    Parameters
    ----------
    folder_path : str
        Path to the TDT tank folder containing ``.tsq`` and ``.tev`` files.
    """

    @classmethod
    def discover_events_and_flags(cls, folder_path: str) -> tuple[list[str], list[str]]:
        """
        Discover available events and format flags from TDT files.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing TDT files.

        Returns
        -------
        events : list of str
            Names of all events/stores available in the dataset.
        flags : list of str
            Format indicators or file type flags.
        """
        header_df, _ = cls._readtsq(folder_path)

        # Populate events from header_df
        if isinstance(header_df, pd.DataFrame):
            header_df["name"] = np.asarray(header_df["name"], dtype=str)
            allnames = np.unique(header_df["name"])
            short_name_indices = []
            for i in range(len(allnames)):
                length = len(str(allnames[i]))
                if length < 4:
                    short_name_indices.append(i)
            allnames = np.delete(allnames, short_name_indices, 0)
            # Epoc stores that encode multiple behaviours are split into one sub-event per
            # unique marker value here (at discover time) so storesList is fully settled
            # before step 2; read() then needs no storesList mutation.
            split_map = cls._compute_split_map(header_df)
            events = []
            for name in allnames:
                if name in split_map:
                    events.extend(split_name for split_name, _ in split_map[name])
                else:
                    events.append(name)
        else:
            events = []

        flags = []
        return events, flags

    def __init__(self, folder_path: str) -> None:
        self.folder_path = folder_path
        self._header_df, _ = self._readtsq(folder_path)

    @staticmethod
    def _readtsq(folder_path: str) -> tuple[pd.DataFrame | int, str | int]:
        logger.debug("Trying to read tsq file.")
        names = ("size", "type", "name", "chan", "sort_code", "timestamp", "fp_loc", "strobe", "format", "frequency")
        formats = (int32, int32, "S4", uint16, uint16, float64, int64, float64, int32, float32)
        offsets = 0, 4, 8, 12, 14, 16, 24, 24, 32, 36
        tsq_dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets}, align=True)
        path = glob.glob(os.path.join(folder_path, "*.tsq"))
        if len(path) > 1:
            message = (
                f"Multiple .tsq files found in '{folder_path}': {sorted(path)}. "
                "Each TDT tank folder must contain exactly one .tsq file."
            )
            logger.error(message)
            raise ValueError(message)
        elif len(path) == 0:
            logger.info("tsq file not found.")
            return 0, 0
        else:
            path = path[0]
            flag = "tsq"

        # reading tsq file
        tsq = np.fromfile(path, dtype=tsq_dtype)

        # creating dataframe of the data
        df = pd.DataFrame(tsq)

        logger.info("Data from tsq file fetched.")
        return df, flag

    def _readtev(self, event: str) -> dict[str, object]:
        header_df = self._header_df.copy()
        folder_path = self.folder_path

        logger.debug("Reading data for event {} ...".format(event))
        tevfilepath = glob.glob(os.path.join(folder_path, "*.tev"))
        if len(tevfilepath) > 1:
            raise ValueError(
                f"Multiple .tev files found in '{folder_path}': {sorted(tevfilepath)}. "
                "Each TDT tank folder must contain exactly one .tev file."
            )
        else:
            tevfilepath = tevfilepath[0]

        header_df["name"] = np.asarray(header_df["name"], dtype=str)

        allnames = np.unique(header_df["name"])

        short_name_indices = []
        for i in range(len(allnames)):
            length = len(str(allnames[i]))
            if length < 4:
                short_name_indices.append(i)

        allnames = np.delete(allnames, short_name_indices, 0)

        eventNew = np.array(list(event))

        name_matches = self._ismember(header_df["name"], event)

        if sum(name_matches) == 0:
            available = sorted(str(name) for name in allnames)
            message = (
                f"Requested TDT store name '{event}' not found in tank "
                f"'{self.folder_path}' (case-sensitive). Available stores: {available}."
            )
            logger.error(message)
            raise ValueError(message)

        allIndexesWhereEventIsPresent = np.where(name_matches == 1)
        first_row = allIndexesWhereEventIsPresent[0][0]

        formatNew = header_df["format"][first_row] + 1

        table = np.array(
            [
                [0, 0, 0, 0],
                [0, "float", 1, np.float32],
                [0, "long", 1, np.int32],
                [0, "short", 2, np.int16],
                [0, "byte", 4, np.int8],
            ]
        )

        event_dict = dict()

        event_dict["storename"] = str(event)
        event_dict["sampling_rate"] = header_df["frequency"][first_row]
        event_dict["timestamps"] = np.asarray(header_df["timestamp"][allIndexesWhereEventIsPresent[0]])
        event_dict["channels"] = np.asarray(header_df["chan"][allIndexesWhereEventIsPresent[0]])

        fp_loc = np.asarray(header_df["fp_loc"][allIndexesWhereEventIsPresent[0]])
        data_size = np.asarray(header_df["size"])

        if formatNew != 5:
            nsample = (data_size[first_row,] - 10) * int(table[formatNew, 2])
            event_dict["data"] = np.zeros((len(fp_loc), nsample))
            for i in range(0, len(fp_loc)):
                with open(tevfilepath, "rb") as tev_file:
                    tev_file.seek(fp_loc[i], os.SEEK_SET)
                    event_dict["data"][i, :] = np.fromfile(tev_file, dtype=table[formatNew, 3], count=nsample).reshape(
                        1, nsample, order="F"
                    )
                    # event_dict['data'] = event_dict['data'].swapaxes()
            event_dict["npoints"] = nsample
        else:
            event_dict["data"] = np.asarray(header_df["strobe"][allIndexesWhereEventIsPresent[0]])
            event_dict["npoints"] = 1
            event_dict["channels"] = np.tile(1, (event_dict["data"].shape[0],))

        event_dict["data"] = (event_dict["data"].T).reshape(-1, order="F")

        return event_dict

    def count_samples(self, *, event: str) -> int:
        """Return the number of samples in ``event`` based on header metadata.

        Split sub-events are counted from the parent epoc store's header rows whose
        strobe value matches.
        """
        header_df = self._header_df
        names = np.asarray(header_df["name"], dtype=str)

        split_lookup = {
            split_name: (parent, value)
            for parent, entries in self._compute_split_map(header_df).items()
            for split_name, value in entries
        }
        if event in split_lookup:
            parent, value = split_lookup[event]
            parent_strobes = np.asarray(header_df["strobe"])[np.where(names == parent)[0]]
            return int(np.count_nonzero(parent_strobes == value))

        name_matches = self._ismember(names, event)
        if sum(name_matches) == 0:
            return 0
        indexes = np.where(name_matches == 1)[0]
        first_row = indexes[0]
        format_new = header_df["format"][first_row] + 1
        if format_new == 5:
            return int(len(indexes))
        bytes_per_sample_table = {1: 1, 2: 1, 3: 2, 4: 4}
        nsample = int((header_df["size"][first_row] - 10) * bytes_per_sample_table[int(format_new)])
        return int(nsample * len(indexes))

    def read(self, *, events: list[str], outputPath: str) -> list[dict[str, Any]]:
        """
        Read data from TDT TEV/TSQ files for the specified events.

        Split sub-events (one per unique marker value of an epoc store that
        encodes multiple behaviours) are enumerated up front by
        :meth:`discover_events_and_flags`, so they appear in ``storesList.csv``
        before this method runs. Here a requested split sub-event is produced by
        reading its parent epoc store once and keeping the timestamps whose
        strobe value matches. This method writes nothing.

        Parameters
        ----------
        events : list of str
            TDT store names to read (case-sensitive). May include split
            sub-event names (e.g. ``"PrtN5"``).
        outputPath : str
            Path to the output directory (unused; required by the base-class
            interface).

        Returns
        -------
        list of dict
            One dictionary per requested event. Each dict contains
            ``storename``, ``sampling_rate``, ``timestamps``, ``data``,
            ``npoints``, and ``channels``.
        """
        split_lookup = {
            split_name: (parent, value)
            for parent, entries in self._compute_split_map(self._header_df).items()
            for split_name, value in entries
        }

        output_dicts = []
        parent_cache: dict[str, dict[str, object]] = {}
        for event in events:
            if event in split_lookup:
                parent, value = split_lookup[event]
                if parent not in parent_cache:
                    parent_cache[parent] = self._readtev(event=parent)
                parent_dict = parent_cache[parent]
                matching_indices = np.where(parent_dict["data"] == value)[0]
                output_dicts.append(
                    {
                        "storename": event,
                        "timestamps": parent_dict["timestamps"][matching_indices],
                        "sampling_rate": parent_dict["sampling_rate"],
                        "data": parent_dict["data"],
                        "npoints": parent_dict["npoints"],
                        "channels": parent_dict["channels"],
                    }
                )
            else:
                output_dicts.append(self._readtev(event=event))
        return output_dicts

    @staticmethod
    def _ismember(values: np.ndarray, element: str) -> np.ndarray:
        membership_flags = [1 if value == element else 0 for value in values]
        return np.asarray(membership_flags)

    @staticmethod
    def _format_split_suffix(value: float) -> str:
        # "{:g}" renders integer-valued codes (int or 5.0) as "5", and collapses
        # float32 precision artifacts (0.10000000149... → "0.1") at 6 significant
        # digits. "." → "p" keeps the suffix free of the "_" / "." delimiters used
        # by downstream filename parsers.
        return f"{float(value):g}".replace(".", "p")

    @staticmethod
    def _event_needs_splitting(data: np.ndarray, sampling_rate: float) -> bool:
        logger.info("Checking event storename data for creating multiple event names from single event storename...")
        diff = np.diff(data)
        if diff.shape[0] == 0:
            return False
        if sampling_rate == 0 and not (np.all(diff == diff[0])):
            return True
        return False

    @classmethod
    def _compute_split_map(cls, header_df: pd.DataFrame | int) -> dict[str, list[tuple[str, float]]]:
        """
        Determine which epoc stores split into multiple sub-events, from header metadata.

        For each epoc/marker store (``format == 4``) whose strobe values encode
        more than one behaviour, returns the per-value split sub-event names.
        This reads only the ``.tsq`` header (the strobe values live there), so
        it can run at discover time without touching the ``.tev`` data.

        Parameters
        ----------
        header_df : pd.DataFrame or int
            The parsed ``.tsq`` header (or ``0`` when no ``.tsq`` is present).

        Returns
        -------
        dict
            Maps each splitting parent store name to a list of
            ``(split_sub_event_name, marker_value)`` pairs, ordered by marker value.
        """
        if not isinstance(header_df, pd.DataFrame):
            return {}

        names = np.asarray(header_df["name"], dtype=str)
        formats = np.asarray(header_df["format"])
        frequencies = np.asarray(header_df["frequency"])
        strobes = np.asarray(header_df["strobe"])

        split_map: dict[str, list[tuple[str, float]]] = {}
        for store in np.unique(names):
            if len(str(store)) < 4:
                continue
            indexes = np.where(names == store)[0]
            first_row = indexes[0]
            # format == 4 (format_new == 5) marks an epoc/marker store; only those carry
            # meaningful strobe values (other stores reuse those bytes as a file pointer).
            if formats[first_row] + 1 != 5:
                continue
            strobe_values = strobes[indexes]
            if not cls._event_needs_splitting(data=strobe_values, sampling_rate=frequencies[first_row]):
                continue
            # Mirror the historical sub-event naming: strip "/" from the store name
            # and append the formatted marker value (e.g. "PAB/" + 16 -> "PAB16").
            new_store = store.replace("/", "")
            split_map[store] = [
                (new_store + cls._format_split_suffix(value), value) for value in np.unique(strobe_values)
            ]
        return split_map

    def _save_dict_to_hdf5(self, event_dict: dict[str, object], outputPath: str) -> None:
        event = event_dict["storename"]
        write_hdf5(event_dict["storename"], event, outputPath, "storename")
        write_hdf5(event_dict["sampling_rate"], event, outputPath, "sampling_rate")
        write_hdf5(event_dict["timestamps"], event, outputPath, "timestamps")
        write_hdf5(event_dict["data"], event, outputPath, "data")
        write_hdf5(event_dict["npoints"], event, outputPath, "npoints")
        write_hdf5(event_dict["channels"], event, outputPath, "channels")

    def save(self, *, output_dicts: list[dict[str, Any]], outputPath: str) -> None:
        """
        Save extracted TDT data dictionaries to HDF5 files.

        Parameters
        ----------
        output_dicts : list of dict
            Data dictionaries as returned by :meth:`read`.
        outputPath : str
            Path to the output directory where HDF5 files are written.
        """
        for event_dict in output_dicts:
            self._save_dict_to_hdf5(event_dict=event_dict, outputPath=outputPath)

    @staticmethod
    def _stub_tev_file(
        tev_file_path: Path | str,
        header_df: pd.DataFrame,
        stubbed_tev_file_path: Path | str,
        stream_name_to_num_segments: dict[str, int],
    ) -> dict[int, int]:
        """
        Write a truncated TEV file containing only the first N data segments per stream.

        Parameters
        ----------
        tev_file_path : Path or str
            Path to the source TEV binary data file.
        header_df : pd.DataFrame
            TSQ header DataFrame as returned by _readtsq (already loaded).
        stubbed_tev_file_path : Path or str
            Destination path for the stubbed TEV file.
        stream_name_to_num_segments : dict
            Mapping of stream name (str) to number of segments to retain.

        Returns
        -------
        original_to_new_fp_loc : dict
            Mapping from original file-pointer positions to their new positions in the stubbed TEV.
        """
        stream_names_bytes = {name.encode() for name in stream_name_to_num_segments}
        with open(tev_file_path, "r+b") as file:
            content = file.read()
        if os.path.exists(stubbed_tev_file_path):
            os.remove(stubbed_tev_file_path)

        all_starts, all_stops, all_stream_names = [], [], []
        for stream_name_bytes in stream_names_bytes:
            row_mask = header_df["name"] == stream_name_bytes
            indexes = np.where(row_mask)[0]
            first_row = indexes[0]
            number_of_samples = header_df["size"][first_row] - 10
            file_positions = np.asarray(header_df["fp_loc"][indexes])
            for position in file_positions:
                all_starts.append(position)
                all_stops.append(position + number_of_samples * 4)
                all_stream_names.append(stream_name_bytes)

        sort_index = np.argsort(all_starts)
        all_starts = np.array(all_starts)[sort_index]
        all_stops = np.array(all_stops)[sort_index]
        all_stream_names = np.array(all_stream_names)[sort_index]

        previous_stop = 0
        write_position = 0
        original_to_new_fp_loc = {}
        stream_name_to_num_written = {name.encode(): 0 for name in stream_name_to_num_segments}
        for start, stop, stream_name_bytes in zip(all_starts, all_stops, all_stream_names):
            with open(stubbed_tev_file_path, "a+b") as file:
                gap = content[previous_stop:start]
                file.write(gap)
                write_position += len(gap)
                number_written = stream_name_to_num_written[stream_name_bytes]
                number_of_segments = stream_name_to_num_segments[stream_name_bytes.decode()]
                if number_written < number_of_segments:
                    original_to_new_fp_loc[start] = write_position
                    segment = content[start:stop]
                    file.write(segment)
                    write_position += len(segment)
                    stream_name_to_num_written[stream_name_bytes] += 1
            previous_stop = stop
        with open(stubbed_tev_file_path, "a+b") as file:
            file.write(content[previous_stop:])
        return original_to_new_fp_loc

    @staticmethod
    def _stub_tsq_file(
        header_df: pd.DataFrame,
        stubbed_tsq_file_path: Path | str,
        stream_name_to_num_segments: dict[str, int],
        original_to_new_fp_loc: dict[int, int],
        stub_duration_in_seconds: float,
    ) -> None:
        """
        Write a truncated TSQ header file matching the stubbed TEV file.

        Parameters
        ----------
        header_df : pd.DataFrame
            TSQ header DataFrame as returned by _readtsq (already loaded).
        stubbed_tsq_file_path : Path or str
            Destination path for the stubbed TSQ file.
        stream_name_to_num_segments : dict
            Mapping of stream name (str) to number of segments to retain.
        original_to_new_fp_loc : dict
            Mapping from original file-pointer positions to their new positions, as returned by _stub_tev_file.
        stub_duration_in_seconds : float
            Desired duration of retained data in seconds. TTL/epoc events beyond this time are excluded.
        """
        names = ("size", "type", "name", "chan", "sort_code", "timestamp", "fp_loc", "strobe", "format", "frequency")
        formats = (int32, int32, "S4", uint16, uint16, float64, int64, float64, int32, float32)
        offsets = 0, 4, 8, 12, 14, 16, 24, 24, 32, 36
        tsq_dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets}, align=True)

        # Reconstruct numpy structured array from the already-loaded DataFrame (avoids re-reading from disk).
        # fp_loc and strobe overlap at offset 24; writing fp_loc covers both.
        tsq_data = np.zeros(len(header_df), dtype=tsq_dtype)
        for field_name in ("size", "type", "name", "chan", "sort_code", "timestamp", "fp_loc", "format", "frequency"):
            tsq_data[field_name] = header_df[field_name].values

        stream_names_bytes = {name.encode() for name in stream_name_to_num_segments}

        # TDT timestamps are Unix epoch seconds, not session-relative.
        # Compute the TTL cutoff as the first continuous stream timestamp plus the desired stub duration.
        first_continuous_timestamp = header_df[header_df["frequency"] > 0]["timestamp"].min()
        ttl_cutoff_timestamp = first_continuous_timestamp + stub_duration_in_seconds

        rows_to_keep = []
        stream_name_to_num_kept = {name.encode(): 0 for name in stream_name_to_num_segments}
        for row in tsq_data:
            stream_name = row["name"]
            if stream_name not in stream_names_bytes:
                is_sentinel = len(stream_name.rstrip(b"\x00")) < 4
                rows_to_keep.append(is_sentinel or row["timestamp"] <= ttl_cutoff_timestamp)
                continue
            number_kept = stream_name_to_num_kept[stream_name]
            number_of_segments = stream_name_to_num_segments[stream_name.decode()]
            if number_kept < number_of_segments:
                rows_to_keep.append(True)
                stream_name_to_num_kept[stream_name] += 1
            else:
                rows_to_keep.append(False)
        rows_to_keep = np.array(rows_to_keep)
        stubbed_tsq_data = tsq_data[rows_to_keep]

        # correct timestamps: last metadata record should match the last stream timestamp
        first_stream_name_bytes = next(iter(stream_names_bytes))
        stubbed_last_timestamp = stubbed_tsq_data["timestamp"][stubbed_tsq_data["name"] == first_stream_name_bytes][-1]
        stubbed_tsq_data["timestamp"][stubbed_tsq_data["name"] == b"\x02"] = stubbed_last_timestamp

        # correct fp_loc: positions shifted because removed segments shrank the TEV file
        stream_mask = np.isin(stubbed_tsq_data["name"], list(stream_names_bytes))
        stubbed_tsq_data["fp_loc"][stream_mask] = [
            original_to_new_fp_loc[location] for location in stubbed_tsq_data["fp_loc"][stream_mask]
        ]

        # correct size: first record's size field encodes total file size in bytes
        record_size = tsq_dtype.itemsize
        stubbed_tsq_data["size"][0] = len(stubbed_tsq_data) * record_size

        stubbed_tsq_data.tofile(stubbed_tsq_file_path)

    @staticmethod
    def _compute_stream_name_to_num_segments(
        header_df: pd.DataFrame, stub_duration_in_seconds: float
    ) -> dict[str, int]:
        """
        Compute the number of TEV segments to retain per stream for a given stub duration.

        Only continuous data streams (sampling_rate > 0) are included. TTL/epoc streams
        and sentinel records (sampling_rate == 0) are excluded; their rows will be
        preserved in full by the stub helpers.

        Parameters
        ----------
        header_df : pd.DataFrame
            TSQ header DataFrame as returned by _readtsq.
        stub_duration_in_seconds : float
            Desired duration of retained data in seconds.

        Returns
        -------
        stream_name_to_num_segments : dict
            Mapping of stream name (str) to number of segments to retain.
        """
        stream_name_to_num_segments = {}
        for stream_name_bytes in header_df["name"].unique():
            first_row_index = header_df.index[header_df["name"] == stream_name_bytes][0]
            sampling_rate = header_df["frequency"][first_row_index]
            if sampling_rate == 0:
                continue
            samples_per_segment = header_df["size"][first_row_index] - 10
            number_of_segments = max(1, math.ceil(stub_duration_in_seconds * sampling_rate / samples_per_segment))
            stream_name_to_num_segments[stream_name_bytes.decode()] = number_of_segments
        return stream_name_to_num_segments

    def stub(self, *, folder_path: str | Path, duration_in_seconds: float = 1.0) -> None:
        """
        Create a stubbed copy of the TDT tank folder with truncated TEV and TSQ files.

        Copies the entire tank folder to `folder_path`, then replaces the TEV and
        TSQ binary files with truncated versions that retain only the first N data segments
        per continuous stream. The number of segments retained per stream is computed
        automatically from `duration_in_seconds` and each stream's sampling rate.
        TTL/epoc streams (sampling_rate == 0) are trimmed to `duration_in_seconds`.

        Parameters
        ----------
        folder_path : str or Path
            Destination directory for the stubbed tank. Created if it does not exist;
            overwritten if it already exists.
        duration_in_seconds : float, optional
            Approximate duration of data to retain in seconds. Default is 1.0.
        """
        stream_name_to_num_segments = self._compute_stream_name_to_num_segments(
            header_df=self._header_df,
            stub_duration_in_seconds=duration_in_seconds,
        )
        folder_path = Path(folder_path)
        source_folder_path = Path(self.folder_path)

        if folder_path.exists():
            shutil.rmtree(folder_path)
        shutil.copytree(source_folder_path, folder_path)

        tev_file_path = glob.glob(os.path.join(self.folder_path, "*.tev"))[0]
        tsq_file_path = glob.glob(os.path.join(self.folder_path, "*.tsq"))[0]

        stubbed_tev_file_path = folder_path / Path(tev_file_path).name
        stubbed_tsq_file_path = folder_path / Path(tsq_file_path).name

        # Remove originals from stub folder so we can write the truncated replacements
        os.remove(stubbed_tev_file_path)
        os.remove(stubbed_tsq_file_path)

        original_to_new_fp_loc = self._stub_tev_file(
            tev_file_path=tev_file_path,
            header_df=self._header_df,
            stubbed_tev_file_path=stubbed_tev_file_path,
            stream_name_to_num_segments=stream_name_to_num_segments,
        )
        self._stub_tsq_file(
            header_df=self._header_df,
            stubbed_tsq_file_path=stubbed_tsq_file_path,
            stream_name_to_num_segments=stream_name_to_num_segments,
            original_to_new_fp_loc=original_to_new_fp_loc,
            stub_duration_in_seconds=duration_in_seconds,
        )
