import glob
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy import float32, float64, int32, int64, uint16
from tqdm import tqdm

from guppy.extractors import BaseRecordingExtractor

logger = logging.getLogger(__name__)


class TdtRecordingExtractor(BaseRecordingExtractor):

    @classmethod
    def discover_events_and_flags(cls, folder_path) -> tuple[list[str], list[str]]:
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
            index = []
            for i in range(len(allnames)):
                length = len(str(allnames[i]))
                if length < 4:
                    index.append(i)
            allnames = np.delete(allnames, index, 0)
            events = list(allnames)
        else:
            events = []

        flags = []
        return events, flags

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self._header_df, _ = self._readtsq(folder_path)

    @staticmethod
    def _readtsq(folder_path):
        logger.debug("Trying to read tsq file.")
        names = ("size", "type", "name", "chan", "sort_code", "timestamp", "fp_loc", "strobe", "format", "frequency")
        formats = (int32, int32, "S4", uint16, uint16, float64, int64, float64, int32, float32)
        offsets = 0, 4, 8, 12, 14, 16, 24, 24, 32, 36
        tsq_dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets}, align=True)
        path = glob.glob(os.path.join(folder_path, "*.tsq"))
        if len(path) > 1:
            logger.error("Two tsq files are present at the location.")
            raise Exception("Two tsq files are present at the location.")
        elif len(path) == 0:
            logger.info("\033[1m" + "tsq file not found." + "\033[1m")
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

    def _readtev(self, event):
        data = self._header_df.copy()
        filepath = self.folder_path

        logger.debug("Reading data for event {} ...".format(event))
        tevfilepath = glob.glob(os.path.join(filepath, "*.tev"))
        if len(tevfilepath) > 1:
            raise Exception("Two tev files are present at the location.")
        else:
            tevfilepath = tevfilepath[0]

        data["name"] = np.asarray(data["name"], dtype=str)

        allnames = np.unique(data["name"])

        index = []
        for i in range(len(allnames)):
            length = len(str(allnames[i]))
            if length < 4:
                index.append(i)

        allnames = np.delete(allnames, index, 0)

        eventNew = np.array(list(event))

        # logger.info(allnames)
        # logger.info(eventNew)
        row = self._ismember(data["name"], event)

        if sum(row) == 0:
            logger.error("\033[1m" + "Requested store name " + event + " not found (case-sensitive)." + "\033[0m")
            logger.error("\033[1m" + "File contains the following TDT store names:" + "\033[0m")
            logger.error("\033[1m" + str(allnames) + "\033[0m")
            logger.error("\033[1m" + "TDT store name " + str(event) + " not found." + "\033[0m")
            raise ValueError("Requested store name not found.")

        allIndexesWhereEventIsPresent = np.where(row == 1)
        first_row = allIndexesWhereEventIsPresent[0][0]

        formatNew = data["format"][first_row] + 1

        table = np.array(
            [
                [0, 0, 0, 0],
                [0, "float", 1, np.float32],
                [0, "long", 1, np.int32],
                [0, "short", 2, np.int16],
                [0, "byte", 4, np.int8],
            ]
        )

        S = dict()

        S["storename"] = str(event)
        S["sampling_rate"] = data["frequency"][first_row]
        S["timestamps"] = np.asarray(data["timestamp"][allIndexesWhereEventIsPresent[0]])
        S["channels"] = np.asarray(data["chan"][allIndexesWhereEventIsPresent[0]])

        fp_loc = np.asarray(data["fp_loc"][allIndexesWhereEventIsPresent[0]])
        data_size = np.asarray(data["size"])

        if formatNew != 5:
            nsample = (data_size[first_row,] - 10) * int(table[formatNew, 2])
            S["data"] = np.zeros((len(fp_loc), nsample))
            for i in range(0, len(fp_loc)):
                with open(tevfilepath, "rb") as fp:
                    fp.seek(fp_loc[i], os.SEEK_SET)
                    S["data"][i, :] = np.fromfile(fp, dtype=table[formatNew, 3], count=nsample).reshape(
                        1, nsample, order="F"
                    )
                    # S['data'] = S['data'].swapaxes()
            S["npoints"] = nsample
        else:
            S["data"] = np.asarray(data["strobe"][allIndexesWhereEventIsPresent[0]])
            S["npoints"] = 1
            S["channels"] = np.tile(1, (S["data"].shape[0],))

        S["data"] = (S["data"].T).reshape(-1, order="F")

        return S

    def read(self, *, events: list[str], outputPath: str) -> list[dict[str, Any]]:
        output_dicts = []
        for event in events:
            S = self._readtev(event=event)
            if self._event_needs_splitting(data=S["data"], sampling_rate=S["sampling_rate"]):
                event_dicts = self._split_event_data(S, event)
                self._split_event_storesList(S, event, outputPath)
            else:
                event_dicts = [S]
            output_dicts.extend(event_dicts)
        return output_dicts

    @staticmethod
    def _ismember(arr, element):
        res = [1 if i == element else 0 for i in arr]
        return np.asarray(res)

    @staticmethod
    def _event_needs_splitting(data, sampling_rate):
        logger.info("Checking event storename data for creating multiple event names from single event storename...")
        diff = np.diff(data)
        if diff.shape[0] == 0:
            return False
        if sampling_rate == 0 and not (np.all(diff == diff[0])):
            return True
        return False

    def _split_event_data(self, S, event):
        # Note that new_event is only used for the new storesList and event is still used for the old storesList
        new_event = event.replace("\\", "")
        new_event = event.replace("/", "")
        logger.info("\033[1m" + "Data in event {} belongs to multiple behavior".format(event) + "\033[0m")
        logger.debug("\033[1m" + "Create timestamp files for individual new event." + "\033[0m")
        i_d = np.unique(S["data"])
        event_dicts = [S]
        for i in range(i_d.shape[0]):
            new_S = dict()
            idx = np.where(S["data"] == i_d[i])[0]
            new_S["timestamps"] = S["timestamps"][idx]
            new_S["storename"] = new_event + str(int(i_d[i]))
            new_S["sampling_rate"] = S["sampling_rate"]
            new_S["data"] = S["data"]
            new_S["npoints"] = S["npoints"]
            new_S["channels"] = S["channels"]
            event_dicts.append(new_S)
        logger.info("\033[1m Timestamp files for individual new event are created.\033[0m")

        return event_dicts

    def _split_event_storesList(self, S, event, outputPath):
        # Note that new_event is only used for the new storesList and event is still used for the old storesList
        new_event = event.replace("\\", "")
        new_event = event.replace("/", "")
        storesList = np.genfromtxt(os.path.join(outputPath, "storesList.csv"), dtype="str", delimiter=",").reshape(
            2, -1
        )
        logger.info("\033[1m" + "StoresList in event {} belongs to multiple behavior".format(event) + "\033[0m")
        logger.debug("\033[1m" + "Change the stores list file for individual new event." + "\033[0m")
        i_d = np.unique(S["data"])
        for i in range(i_d.shape[0]):
            storesList = np.concatenate(
                (storesList, [[new_event + str(int(i_d[i]))], [new_event + "_" + str(int(i_d[i]))]]), axis=1
            )

        idx = np.where(storesList[0] == event)[0]
        storesList = np.delete(storesList, idx, axis=1)
        if not os.path.exists(os.path.join(outputPath, ".cache_storesList.csv")):
            os.rename(os.path.join(outputPath, "storesList.csv"), os.path.join(outputPath, ".cache_storesList.csv"))
        if idx.shape[0] == 0:
            pass
        else:
            np.savetxt(os.path.join(outputPath, "storesList.csv"), storesList, delimiter=",", fmt="%s")
        logger.info("\033[1m The stores list file is changed.\033[0m")

    def _save_dict_to_hdf5(self, S, outputPath):
        event = S["storename"]
        self._write_hdf5(S["storename"], event, outputPath, "storename")
        self._write_hdf5(S["sampling_rate"], event, outputPath, "sampling_rate")
        self._write_hdf5(S["timestamps"], event, outputPath, "timestamps")
        self._write_hdf5(S["data"], event, outputPath, "data")
        self._write_hdf5(S["npoints"], event, outputPath, "npoints")
        self._write_hdf5(S["channels"], event, outputPath, "channels")

    def save(self, *, output_dicts: list[dict[str, Any]], outputPath: str) -> None:
        for S in output_dicts:
            self._save_dict_to_hdf5(S=S, outputPath=outputPath)

    @staticmethod
    def _stub_tev_file(tev_file_path, header_df, stubbed_tev_file_path, stream_name_to_num_segments):
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
    def _stub_tsq_file(header_df, stubbed_tsq_file_path, stream_name_to_num_segments, original_to_new_fp_loc):
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

        rows_to_keep = []
        stream_name_to_num_kept = {name.encode(): 0 for name in stream_name_to_num_segments}
        for row in tqdm(tsq_data):
            stream_name = row["name"]
            if stream_name not in stream_names_bytes:
                rows_to_keep.append(True)
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

    def stub(self, *, stub_folder_path, stream_name_to_num_segments):
        """
        Create a stubbed copy of the TDT tank folder with truncated TEV and TSQ files.

        Copies the entire tank folder to `stub_folder_path`, then replaces the TEV and
        TSQ binary files with truncated versions that retain only the first N data segments
        per stream. All other files (metadata, text listings, etc.) are copied unchanged.

        Parameters
        ----------
        stub_folder_path : str or Path
            Destination directory for the stubbed tank. Created if it does not exist;
            overwritten if it already exists.
        stream_name_to_num_segments : dict
            Mapping of TDT store name (str, e.g. ``"Dv1A"``) to the number of data
            segments to retain in the stubbed files.
        """
        stub_folder_path = Path(stub_folder_path)
        source_folder_path = Path(self.folder_path)

        if stub_folder_path.exists():
            shutil.rmtree(stub_folder_path)
        shutil.copytree(source_folder_path, stub_folder_path)

        tev_file_path = glob.glob(os.path.join(self.folder_path, "*.tev"))[0]
        tsq_file_path = glob.glob(os.path.join(self.folder_path, "*.tsq"))[0]

        stubbed_tev_file_path = stub_folder_path / Path(tev_file_path).name
        stubbed_tsq_file_path = stub_folder_path / Path(tsq_file_path).name

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
        )
