import logging
import tkinter as tk
from tkinter import StringVar, messagebox, ttk

logger = logging.getLogger(__name__)


def get_multi_event_responses(multiple_event_ttls):
    responses = []
    for has_multiple in multiple_event_ttls:
        if not has_multiple:
            responses.append(False)
            continue
        window = tk.Tk()
        response = messagebox.askyesno(
            "Multiple event TTLs",
            (
                "Based on the TTL file, "
                "it looks like TTLs "
                "belong to multiple behavior types. "
                "Do you want to create multiple files for each "
                "behavior type?"
            ),
        )
        window.destroy()
        responses.append(response)
    return responses


def get_timestamp_configuration(ts_unit_needs, col_names_ts):
    ts_units, npm_timestamp_column_names = [], []
    for need in ts_unit_needs:
        if not need:
            ts_units.append("seconds")
            npm_timestamp_column_names.append(None)
            continue
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
            window,
            values=["", "seconds", "milliseconds", "microseconds"],
            textvariable=holdComboboxValues["time_unit"],
        )
        time_unit_combo.grid(row=1, column=2, pady=25, padx=25)
        time_unit_combo.current(0)
        # time_unit_combo.bind("<<ComboboxSelected>>", comboBoxSelected)
        window.lift()
        window.after(500, lambda: window.lift())
        window.mainloop()

        if holdComboboxValues["timestamps"].get():
            npm_timestamp_column_name = holdComboboxValues["timestamps"].get()
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
        ts_units.append(ts_unit)
        npm_timestamp_column_names.append(npm_timestamp_column_name)
    return ts_units, npm_timestamp_column_names
