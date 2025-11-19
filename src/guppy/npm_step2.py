import glob
import logging
import os
import tkinter as tk
from tkinter import StringVar, messagebox, ttk

import numpy as np
import pandas as pd
import panel as pn

pn.extension()

logger = logging.getLogger(__name__)


def import_npm(filepath, num_ch, inputParameters=None):

    logger.debug("If it exists, importing NPM file based on the structure of file")
    # Headless configuration (used to avoid any UI prompts when running tests)
    headless = bool(os.environ.get("GUPPY_BASE_DIR"))
    npm_timestamp_column_name = None
    npm_time_unit = None
    npm_split_events = None
    if isinstance(inputParameters, dict):
        npm_timestamp_column_name = inputParameters.get("npm_timestamp_column_name")
        npm_time_unit = inputParameters.get("npm_time_unit", "seconds")
        npm_split_events = inputParameters.get("npm_split_events", True)
    path = sorted(glob.glob(os.path.join(filepath, "*.csv"))) + sorted(glob.glob(os.path.join(filepath, "*.doric")))
    path_chev = glob.glob(os.path.join(filepath, "*chev*"))
    path_chod = glob.glob(os.path.join(filepath, "*chod*"))
    path_chpr = glob.glob(os.path.join(filepath, "*chpr*"))
    path_event = glob.glob(os.path.join(filepath, "event*"))
    # path_sig = glob.glob(os.path.join(filepath, 'sig*')) # TODO: what is this for?
    path_chev_chod_event = path_chev + path_chod + path_event + path_chpr

    path = sorted(list(set(path) - set(path_chev_chod_event)))
    flag = "None"
    event_from_filename = []
    flag_arr = []
    for i in range(len(path)):
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
        _, value = check_header(df)

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
        elif len(cols) >= 2:
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
            df, indices_dict, _ = decide_indices(file, df, flag, num_ch)
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
            if headless:
                response = 1 if bool(npm_split_events) else 0
            else:
                window = tk.Tk()
                if len(type_val_unique) > 1:
                    response = messagebox.askyesno(
                        "Multiple event TTLs",
                        "Based on the TTL file,\
                                                                        it looks like TTLs \
                                                                        belongs to multiple behavior type. \
                                                                        Do you want to create multiple files for each \
                                                                        behavior type ?",
                    )
                else:
                    response = 0
                window.destroy()
            if response == 1:
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
            df, ts_unit = decide_ts_unit_for_npm(
                df, timestamp_column_name=npm_timestamp_column_name, time_unit=npm_time_unit, headless=headless
            )
            df, indices_dict, _ = decide_indices(file, df, flag)
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

        path_chev = glob.glob(os.path.join(filepath, "*chev*"))
        path_chod = glob.glob(os.path.join(filepath, "*chod*"))
        path_chpr = glob.glob(os.path.join(filepath, "*chpr*"))
        path_event = glob.glob(os.path.join(filepath, "event*"))
        # path_sig = glob.glob(os.path.join(filepath, 'sig*'))
        path_chev_chod_chpr = [path_chev, path_chod, path_chpr]
        if (
            ("data_np_v2" in flag_arr or "data_np" in flag_arr) and ("event_np" in flag_arr) and (i == len(path) - 1)
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

            for j in range(len(path_event)):
                df_event = pd.read_csv(path_event[j])
                df_chev = pd.read_csv(path_chev[0])
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


def check_header(df):
    arr = list(df.columns)
    check_float = []
    for i in arr:
        try:
            check_float.append(float(i))
        except:
            pass

    return arr, check_float


# function to decide indices of interleaved channels
# in neurophotometrics data
def decide_indices(file, df, flag, num_ch=2):
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

        num_ch, ch = check_channels(state)
        indices_dict = dict()
        for i in range(num_ch):
            first_occurrence = np.where(state == ch[i])[0]
            indices_dict[ch_name[i]] = np.arange(first_occurrence[0], df.shape[0], num_ch)

        df = df.drop(arr, axis=1)

    return df, indices_dict, num_ch


# check flag consistency in neurophotometrics data
def check_channels(state):
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


# function to decide NPM timestamps unit (seconds, ms or us)
def decide_ts_unit_for_npm(df, timestamp_column_name=None, time_unit=None, headless=False):
    col_names = np.array(list(df.columns))
    col_names_ts = [""]
    for name in col_names:
        if "timestamp" in name.lower():
            col_names_ts.append(name)

    ts_unit = "seconds"
    if len(col_names_ts) > 2:
        # Headless path: auto-select column/unit without any UI
        if headless:
            if timestamp_column_name is not None:
                assert (
                    timestamp_column_name in col_names_ts
                ), f"Provided timestamp_column_name '{timestamp_column_name}' not found in columns {col_names_ts[1:]}"
                chosen = timestamp_column_name
            else:
                chosen = col_names_ts[1]
            df.insert(1, "Timestamp", df[chosen])
            df = df.drop(col_names_ts[1:], axis=1)
            valid_units = {"seconds", "milliseconds", "microseconds"}
            ts_unit = time_unit if (isinstance(time_unit, str) and time_unit in valid_units) else "seconds"
            return df, ts_unit
        # def comboBoxSelected(event):
        #    logger.info(event.widget.get())

        window = tk.Tk()
        window.title("Select appropriate options for timestamps")
        window.geometry("500x200")
        holdComboboxValues = dict()

        timestamps_label = ttk.Label(window, text="Select which timestamps to use : ").grid(
            row=0, column=1, pady=25, padx=25
        )
        holdComboboxValues["timestamps"] = StringVar()
        timestamps_combo = ttk.Combobox(window, values=col_names_ts, textvariable=holdComboboxValues["timestamps"])
        timestamps_combo.grid(row=0, column=2, pady=25, padx=25)
        timestamps_combo.current(0)
        # timestamps_combo.bind("<<ComboboxSelected>>", comboBoxSelected)

        time_unit_label = ttk.Label(window, text="Select timestamps unit : ").grid(row=1, column=1, pady=25, padx=25)
        holdComboboxValues["time_unit"] = StringVar()
        time_unit_combo = ttk.Combobox(
            window, values=["", "seconds", "milliseconds", "microseconds"], textvariable=holdComboboxValues["time_unit"]
        )
        time_unit_combo.grid(row=1, column=2, pady=25, padx=25)
        time_unit_combo.current(0)
        # time_unit_combo.bind("<<ComboboxSelected>>", comboBoxSelected)
        window.lift()
        window.after(500, lambda: window.lift())
        window.mainloop()

        if holdComboboxValues["timestamps"].get():
            df.insert(1, "Timestamp", df[holdComboboxValues["timestamps"].get()])
            df = df.drop(col_names_ts[1:], axis=1)
        else:
            messagebox.showerror(
                "All options not selected",
                "All the options for timestamps \
                                                            were not selected. Please select appropriate options",
            )
            logger.error(
                "All the options for timestamps \
                        were not selected. Please select appropriate options"
            )
            raise Exception(
                "All the options for timestamps \
                            were not selected. Please select appropriate options"
            )
        if holdComboboxValues["time_unit"].get():
            if holdComboboxValues["time_unit"].get() == "seconds":
                ts_unit = holdComboboxValues["time_unit"].get()
            elif holdComboboxValues["time_unit"].get() == "milliseconds":
                ts_unit = holdComboboxValues["time_unit"].get()
            else:
                ts_unit = holdComboboxValues["time_unit"].get()
        else:
            messagebox.showerror(
                "All options not selected",
                "All the options for timestamps \
                                                            were not selected. Please select appropriate options",
            )
            logger.error(
                "All the options for timestamps \
                        were not selected. Please select appropriate options"
            )
            raise Exception(
                "All the options for timestamps \
                            were not selected. Please select appropriate options"
            )
    else:
        pass

    return df, ts_unit
