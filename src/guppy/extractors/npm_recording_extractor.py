import glob
import logging
import os
import shutil
from pathlib import Path

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
    # Inherits from CsvRecordingExtractor to reuse identical read/save logic.
    # Only overrides discover_events_and_flags() and adds NPM-specific helper methods.

    @classmethod
    def discover_events_and_flags(cls, folder_path, num_ch, inputParameters) -> tuple[list[str], list[str]]:
        """
        Discover available events and format flags from NPM files.

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
        logger.debug("If it exists, importing NPM file based on the structure of file")
        if isinstance(inputParameters, dict):
            npm_timestamp_column_names = inputParameters.get("npm_timestamp_column_names")
            npm_time_units = inputParameters.get("npm_time_units")
            # TODO: come up with a better name for npm_split_events that can be appropriately pluralized for a list
            npm_split_events = inputParameters.get("npm_split_events")
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
        # Exclude single-column timestamp CSVs — those are handled by CsvRecordingExtractor.
        path = [p for p in path if not (p.endswith(".csv") and _is_event_csv(p))]
        flag = "None"
        event_from_filename = []
        flag_arr = []
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

            dirname = os.path.dirname(path[i])
            ext = os.path.basename(path[i]).split(".")[-1]
            assert ext != "doric", "Doric files are not supported by import_npm function."
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
            ), "This file appears to be doric .csv. This function only supports NPM .csv files."
            df = pd.read_csv(path[i], index_col=False)
            _, value = cls._check_header(df)

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
            assert len(cols) != 1, "File appears to be event .csv. This function only supports NPM .csv files."
            assert len(cols) != 3, "File appears to be data .csv. This function only supports NPM .csv files."
            if len(cols) == 2:
                flag = "event_or_data_np"
            elif len(cols) > 2:
                flag = "data_np"
            else:
                logger.error("Number of columns in csv file does not make sense.")
                raise Exception("Number of columns in csv file does not make sense.")

            if columns_isstr == True and (
                "flags" in np.char.lower(np.array(cols)) or "ledstate" in np.char.lower(np.array(cols))
            ):
                flag = flag + "_v2"
            else:
                flag = flag

            # used assigned flags to process the files and read the data
            if flag == "event_or_data_np":
                arr = list(df.iloc[:, 1])
                check_float = [True for i in arr if isinstance(i, float)]
                if len(arr) == len(check_float) and columns_isstr == False:
                    flag = "data_np"
                elif columns_isstr == True and ("value" in np.char.lower(np.array(cols))):
                    flag = "event_np"
                else:
                    flag = "event_np"

            flag_arr.append(flag)
            logger.info(flag)
            if flag == "data_np":
                file = f"file{str(i)}_"
                df, indices_dict, _ = cls.decide_indices(file, df, flag, num_ch)
                keys = list(indices_dict.keys())
                for k in range(len(keys)):
                    for j in range(df.shape[1]):
                        if j == 0:
                            timestamps = df.iloc[:, j][indices_dict[keys[k]]]
                            # timestamps_odd = df.iloc[:,j][odd_indices]
                        else:
                            d = dict()
                            d["timestamps"] = timestamps
                            d["data"] = df.iloc[:, j][indices_dict[keys[k]]]

                            df_ch = pd.DataFrame(d)
                            df_ch.to_csv(os.path.join(dirname, keys[k] + str(j) + ".csv"), index=False)
                            event_from_filename.append(keys[k] + str(j))

            elif flag == "event_np":
                type_val = np.array(df.iloc[:, 1])
                type_val_unique = np.unique(type_val)
                if split_events:
                    timestamps = np.array(df.iloc[:, 0])
                    for j in range(len(type_val_unique)):
                        idx = np.where(type_val == type_val_unique[j])
                        d = dict()
                        d["timestamps"] = timestamps[idx]
                        df_new = pd.DataFrame(d)
                        df_new.to_csv(os.path.join(dirname, "event" + str(type_val_unique[j]) + ".csv"), index=False)
                        event_from_filename.append("event" + str(type_val_unique[j]))
                else:
                    timestamps = np.array(df.iloc[:, 0])
                    d = dict()
                    d["timestamps"] = timestamps
                    df_new = pd.DataFrame(d)
                    df_new.to_csv(os.path.join(dirname, "event" + str(0) + ".csv"), index=False)
                    event_from_filename.append("event" + str(0))
            else:
                file = f"file{str(i)}_"
                ts_unit = npm_time_unit
                df = cls._update_df_with_timestamp_columns(df, timestamp_column_name=npm_timestamp_column_name)
                df, indices_dict, _ = cls.decide_indices(file, df, flag)
                keys = list(indices_dict.keys())
                for k in range(len(keys)):
                    for j in range(df.shape[1]):
                        if j == 0:
                            timestamps = df.iloc[:, j][indices_dict[keys[k]]]
                            # timestamps_odd = df.iloc[:,j][odd_indices]
                        else:
                            d = dict()
                            d["timestamps"] = timestamps
                            d["data"] = df.iloc[:, j][indices_dict[keys[k]]]

                            df_ch = pd.DataFrame(d)
                            df_ch.to_csv(os.path.join(dirname, keys[k] + str(j) + ".csv"), index=False)
                            event_from_filename.append(keys[k] + str(j))

            path_chev = glob.glob(os.path.join(folder_path, "*chev*"))
            path_chod = glob.glob(os.path.join(folder_path, "*chod*"))
            path_chpr = glob.glob(os.path.join(folder_path, "*chpr*"))
            path_event = glob.glob(os.path.join(folder_path, "event*"))
            # path_sig = glob.glob(os.path.join(filepath, 'sig*'))
            path_chev_chod_chpr = [path_chev, path_chod, path_chpr]
            if (
                ("data_np_v2" in flag_arr or "data_np" in flag_arr)
                and ("event_np" in flag_arr)
                and (i == len(path) - 1)
            ) or (
                ("data_np_v2" in flag_arr or "data_np" in flag_arr) and (i == len(path) - 1)
            ):  # i==len(path)-1 and or 'event_np' in flag
                num_path_chev, num_path_chod, num_path_chpr = len(path_chev), len(path_chod), len(path_chpr)
                arr_len, no_ch = [], []
                for i in range(len(path_chev_chod_chpr)):
                    if len(path_chev_chod_chpr[i]) > 0:
                        arr_len.append(len(path_chev_chod_chpr[i]))
                    else:
                        continue

                unique_arr_len = np.unique(np.array(arr_len))
                if "data_np_v2" in flag_arr:
                    if ts_unit == "seconds":
                        divisor = 1
                    elif ts_unit == "milliseconds":
                        divisor = 1e3
                    else:
                        divisor = 1e6
                else:
                    divisor = 1000

                df_chev = pd.read_csv(path_chev[0])
                for j in range(len(path_event)):
                    df_event = pd.read_csv(path_event[j])
                    df_event["timestamps"] = (df_event["timestamps"] - df_chev["timestamps"][0]) / divisor
                    df_event.to_csv(path_event[j], index=False)
                if unique_arr_len.shape[0] == 1:
                    for j in range(len(path_chev)):
                        if file + "chev" in indices_dict.keys():
                            df_chev = pd.read_csv(path_chev[j])
                            df_chev["timestamps"] = (df_chev["timestamps"] - df_chev["timestamps"][0]) / divisor
                            df_chev["sampling_rate"] = np.full(df_chev.shape[0], np.nan)
                            df_chev.at[0, "sampling_rate"] = df_chev.shape[0] / (
                                df_chev["timestamps"].iloc[-1] - df_chev["timestamps"].iloc[0]
                            )
                            df_chev.to_csv(path_chev[j], index=False)

                        if file + "chod" in indices_dict.keys():
                            df_chod = pd.read_csv(path_chod[j])
                            df_chod["timestamps"] = df_chev["timestamps"]
                            df_chod["sampling_rate"] = np.full(df_chod.shape[0], np.nan)
                            df_chod.at[0, "sampling_rate"] = df_chev["sampling_rate"][0]
                            df_chod.to_csv(path_chod[j], index=False)

                        if file + "chpr" in indices_dict.keys():
                            df_chpr = pd.read_csv(path_chpr[j])
                            df_chpr["timestamps"] = df_chev["timestamps"]
                            df_chpr["sampling_rate"] = np.full(df_chpr.shape[0], np.nan)
                            df_chpr.at[0, "sampling_rate"] = df_chev["sampling_rate"][0]
                            df_chpr.to_csv(path_chpr[j], index=False)
                else:
                    logger.error("Number of channels should be same for all regions.")
                    raise Exception("Number of channels should be same for all regions.")
        logger.info("Importing of NPM file is done.")
        return event_from_filename, flag_arr

    def stub(self, *, folder_path, duration_in_seconds=1.0):
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
                    (col for col in dataframe.columns if "timestamp" in col.lower()),
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
    def has_multiple_event_ttls(cls, folder_path):
        path = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
        path_chev = glob.glob(os.path.join(folder_path, "*chev*"))
        path_chod = glob.glob(os.path.join(folder_path, "*chod*"))
        path_chpr = glob.glob(os.path.join(folder_path, "*chpr*"))
        path_event = glob.glob(os.path.join(folder_path, "event*"))
        path_chev_chod_event = path_chev + path_chod + path_event + path_chpr

        path = sorted(list(set(path) - set(path_chev_chod_event)))
        path = [p for p in path if not _is_event_csv(p)]  # Skip event CSVs, which are handled by CsvRecordingExtractor
        multiple_event_ttls = []
        for i in range(len(path)):
            df = pd.read_csv(path[i], index_col=False)
            _, value = cls._check_header(df)

            # check dataframe structure and read data accordingly
            if len(value) > 0:
                columns_isstr = False
                df = pd.read_csv(path[i], header=None)
                cols = np.array(list(df.columns), dtype=str)
            else:
                columns_isstr = True
                cols = np.array(list(df.columns), dtype=str)
            if len(cols) == 2:
                flag = "event_or_data_np"
            elif len(cols) > 2:
                flag = "data_np"
            else:
                logger.error("Number of columns in csv file does not make sense.")
                raise Exception("Number of columns in csv file does not make sense.")

            # used assigned flags to process the files and read the data
            if flag == "event_or_data_np":
                arr = list(df.iloc[:, 1])
                check_float = [True for i in arr if isinstance(i, float)]
                if len(arr) == len(check_float) and columns_isstr == False:
                    flag = "data_np"
                elif columns_isstr == True and ("value" in np.char.lower(np.array(cols))):
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
    def decide_indices(cls, file, df, flag, num_ch=2):
        ch_name = [file + "chev", file + "chod", file + "chpr"]
        if len(ch_name) < num_ch:
            logger.error(
                "Number of channels parameters in Input Parameters GUI is more than 3. \
                        Looks like there are more than 3 channels in the file. Reading of these files\
                        are not supported. Reach out to us if you get this error message."
            )
            raise Exception(
                "Number of channels parameters in Input Parameters GUI is more than 3. \
                            Looks like there are more than 3 channels in the file. Reading of these files\
                            are not supported. Reach out to us if you get this error message."
            )
        if flag == "data_np":
            indices_dict = dict()
            for i in range(num_ch):
                indices_dict[ch_name[i]] = np.arange(i, df.shape[0], num_ch)

        else:
            cols = np.array(list(df.columns))
            if "flags" in np.char.lower(np.array(cols)):
                arr = ["FrameCounter", "Flags"]
                state = np.array(df["Flags"])
            elif "ledstate" in np.char.lower(np.array(cols)):
                arr = ["FrameCounter", "LedState"]
                state = np.array(df["LedState"])
            else:
                logger.error(
                    "File type shows Neurophotometrics newer version \
                        data but column names does not have Flags or LedState"
                )
                raise Exception(
                    "File type shows Neurophotometrics newer version \
                                data but column names does not have Flags or LedState"
                )

            num_ch, ch = cls.check_channels(state)
            indices_dict = dict()
            for i in range(num_ch):
                first_occurrence = np.where(state == ch[i])[0]
                indices_dict[ch_name[i]] = np.arange(first_occurrence[0], df.shape[0], num_ch)

            df = df.drop(arr, axis=1)

        return df, indices_dict, num_ch

    # check flag consistency in neurophotometrics data
    @classmethod
    def check_channels(cls, state):
        state = state.astype(int)
        unique_state = np.unique(state[2:12])
        if unique_state.shape[0] > 3:
            logger.error(
                "Looks like there are more than 3 channels in the file. Reading of these files\
                            are not supported. Reach out to us if you get this error message."
            )
            raise Exception(
                "Looks like there are more than 3 channels in the file. Reading of these files\
                            are not supported. Reach out to us if you get this error message."
            )

        return unique_state.shape[0], unique_state

    @classmethod
    def needs_ts_unit(cls, folder_path, num_ch):
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
        path = [p for p in path if not _is_event_csv(p)]  # Skip event CSVs, which are handled by CsvRecordingExtractor
        ts_unit_needs = []
        col_names_ts = [""]
        for i in range(len(path)):
            df = pd.read_csv(path[i], index_col=False)
            _, value = cls._check_header(df)

            # check dataframe structure and read data accordingly
            if len(value) > 0:
                df = pd.read_csv(path[i], header=None)
                cols = np.array(list(df.columns), dtype=str)
                columns_isstr = False
            else:
                columns_isstr = True
                cols = np.array(list(df.columns), dtype=str)
            # check the structure of dataframe and assign flag to the type of file
            if len(cols) == 2:
                flag = "event_or_data_np"
            elif len(cols) > 2:
                flag = "data_np"
            else:
                logger.error("Number of columns in csv file does not make sense.")
                raise Exception("Number of columns in csv file does not make sense.")

            if columns_isstr == True and (
                "flags" in np.char.lower(np.array(cols)) or "ledstate" in np.char.lower(np.array(cols))
            ):
                flag = flag + "_v2"

            # used assigned flags to process the files and read the data
            if flag == "event_or_data_np":
                arr = list(df.iloc[:, 1])
                check_float = [True for i in arr if isinstance(i, float)]
                if len(arr) == len(check_float) and columns_isstr == False:
                    flag = "data_np"
                elif columns_isstr == True and ("value" in np.char.lower(np.array(cols))):
                    flag = "event_np"
                else:
                    flag = "event_np"

            if flag == "data_np":
                file = f"file{str(i)}_"
                df, _, _ = cls.decide_indices(file, df, flag, num_ch)

            if flag == "event_np" or flag == "data_np":
                ts_unit_needs.append(False)
                continue

            col_names = np.array(list(df.columns))
            for name in col_names:
                if "timestamp" in name.lower():
                    col_names_ts.append(name)

            if len(col_names_ts) > 2:
                ts_unit_needs.append(True)
            else:
                ts_unit_needs.append(False)

        return ts_unit_needs, col_names_ts

    @staticmethod
    def _update_df_with_timestamp_columns(df, timestamp_column_name):
        col_names = np.array(list(df.columns))
        col_names_ts = [""]
        for name in col_names:
            if "timestamp" in name.lower():
                col_names_ts.append(name)
        if len(col_names_ts) <= 2:
            return df

        timestamp_column_name = timestamp_column_name if timestamp_column_name is not None else col_names_ts[1]
        assert (
            timestamp_column_name in col_names_ts
        ), f"Provided timestamp_column_name '{timestamp_column_name}' not found in columns {col_names_ts[1:]}"
        df.insert(1, "Timestamp", df[timestamp_column_name])
        df = df.drop(col_names_ts[1:], axis=1)
        return df
