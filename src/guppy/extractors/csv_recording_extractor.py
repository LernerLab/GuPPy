import glob
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CsvRecordingExtractor:

    def __init__(self, folder_path):
        self.folder_path = folder_path

        logger.debug("If it exists, importing either NPM or Doric or csv file based on the structure of file")
        path = sorted(glob.glob(os.path.join(self.folder_path, "*.csv")))

        path = sorted(list(set(path)))
        flag = "None"
        event_from_filename = []
        flag_arr = []
        for i in range(len(path)):
            ext = os.path.basename(path[i]).split(".")[-1]
            assert ext == "csv", "Only .csv files are supported by import_csv function."
            df = pd.read_csv(path[i], header=None, nrows=2, index_col=False, dtype=str)
            df = df.dropna(axis=1, how="all")
            df_arr = np.array(df).flatten()
            check_all_str = []
            for element in df_arr:
                try:
                    float(element)
                except:
                    check_all_str.append(i)
            assert len(check_all_str) != len(
                df_arr
            ), "This file appears to be doric .csv. This function only supports standard .csv files."
            df = pd.read_csv(path[i], index_col=False)

            _, value = self.check_header(df)

            # check dataframe structure and read data accordingly
            if len(value) > 0:
                columns_isstr = False
                df = pd.read_csv(path[i], header=None)
                cols = np.array(list(df.columns), dtype=str)
            else:
                df = df
                columns_isstr = True
                cols = np.array(list(df.columns), dtype=str)
            # check the structure of dataframe and assign flag to the type of file
            if len(cols) == 1:
                if cols[0].lower() != "timestamps":
                    logger.error("\033[1m" + "Column name should be timestamps (all lower-cases)" + "\033[0m")
                    raise Exception("\033[1m" + "Column name should be timestamps (all lower-cases)" + "\033[0m")
                else:
                    flag = "event_csv"
            elif len(cols) == 3:
                arr1 = np.array(["timestamps", "data", "sampling_rate"])
                arr2 = np.char.lower(np.array(cols))
                if (np.sort(arr1) == np.sort(arr2)).all() == False:
                    logger.error(
                        "\033[1m"
                        + "Column names should be timestamps, data and sampling_rate (all lower-cases)"
                        + "\033[0m"
                    )
                    raise Exception(
                        "\033[1m"
                        + "Column names should be timestamps, data and sampling_rate (all lower-cases)"
                        + "\033[0m"
                    )
                else:
                    flag = "data_csv"
            elif len(cols) == 2:
                raise ValueError(
                    "Data appears to be Neurophotometrics csv. Please use import_npm_csv function to import the data."
                )
            elif len(cols) >= 2:
                raise ValueError(
                    "Data appears to be Neurophotometrics csv. Please use import_npm_csv function to import the data."
                )
            else:
                logger.error("Number of columns in csv file does not make sense.")
                raise Exception("Number of columns in csv file does not make sense.")

            if columns_isstr == True and (
                "flags" in np.char.lower(np.array(cols)) or "ledstate" in np.char.lower(np.array(cols))
            ):
                flag = flag + "_v2"
            else:
                flag = flag

            flag_arr.append(flag)
            logger.info(flag)
            assert (
                flag == "event_csv" or flag == "data_csv"
            ), "This function only supports standard event_csv and data_csv files."
            name = os.path.basename(path[i]).split(".")[0]
            event_from_filename.append(name)

        logger.info("Importing of csv file is done.")

        self.events = event_from_filename
        self.flags = flag_arr

    def check_header(self, df):
        arr = list(df.columns)
        check_float = []
        for i in arr:
            try:
                check_float.append(float(i))
            except:
                pass

        return arr, check_float
