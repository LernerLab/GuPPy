import glob
import logging
import multiprocessing as mp
import os
import time
from itertools import repeat
from typing import Any

import numpy as np
import pandas as pd
from numpy import float32, float64, int32, int64, uint16

from guppy.extractors import BaseRecordingExtractor

logger = logging.getLogger(__name__)


def read_and_save_tdt(extractor, event, outputPath):
    output_dicts = extractor.read(events=[event], outputPath=outputPath)
    extractor.save(output_dicts=output_dicts, outputPath=outputPath)


def execute_readtev(folder_path, events, outputPath, numProcesses=mp.cpu_count()):
    extractor = TdtRecordingExtractor(folder_path=folder_path)
    start = time.time()
    with mp.Pool(numProcesses) as p:
        p.starmap(read_and_save_tdt, zip(repeat(extractor), events, repeat(outputPath)))
    logger.info("Time taken = {0:.5f}".format(time.time() - start))


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
        data = self._header_df
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

    def _ismember(self, arr, element):
        res = [1 if i == element else 0 for i in arr]
        return np.asarray(res)

    def _event_needs_splitting(self, data, sampling_rate):
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
