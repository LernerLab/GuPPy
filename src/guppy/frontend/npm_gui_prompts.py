import logging
import tkinter as tk
from tkinter import StringVar, messagebox, ttk

logger = logging.getLogger(__name__)


def _validate_timestamp_configuration(*, timestamp_column_name: str, time_unit: str) -> None:
    """Raise ValueError if either NPM timestamp combobox value is empty.

    Extracted from ``get_timestamp_configuration`` so the validation logic can be
    unit-tested without spinning up a Tk mainloop.
    """
    missing_fields = []
    if not timestamp_column_name:
        missing_fields.append("'Select which timestamps to use'")
    if not time_unit:
        missing_fields.append("'Select timestamps unit'")
    if missing_fields:
        message = (
            f"NPM timestamp configuration incomplete: {', '.join(missing_fields)} "
            "must be selected before continuing."
        )
        logger.error(message)
        raise ValueError(message)


# get_multi_event_responses is not covered by tests due to flaky behavior of tkinter messagebox in testing environments.
def get_multi_event_responses(multiple_event_ttls):  # pragma: no cover
    """Prompt the user to confirm whether each TTL file contains multiple event types.

    Parameters
    ----------
    multiple_event_ttls : sequence of bool
        One entry per TTL file; ``True`` when the file appears to contain
        multiple event types and a dialog should be shown.

    Returns
    -------
    list of bool
        One boolean per input entry, ``True`` when the user confirmed that the
        corresponding TTL file has multiple event types.
    """
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


# get_timestamp_configuration is not covered by tests due to the use of tkinter GUI elements in the function.
def get_timestamp_configuration(ts_unit_needs, col_names_ts):  # pragma: no cover
    """Prompt the user to select the timestamp column and time unit for each NPM session.

    Parameters
    ----------
    ts_unit_needs : sequence of bool
        One entry per session; ``True`` when a dialog must be shown because the
        timestamp configuration cannot be inferred automatically.
    col_names_ts : sequence of str
        Column names available in the timestamp file, presented as options in
        the combo box.

    Returns
    -------
    ts_units : list of str
        Time unit (``"seconds"``, ``"milliseconds"``, or ``"microseconds"``) for
        each session.
    npm_timestamp_column_names : list of str or None
        Selected timestamp column name for each session, or ``None`` when no
        dialog was needed.
    """
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

        npm_timestamp_column_name = holdComboboxValues["timestamps"].get()
        ts_unit = holdComboboxValues["time_unit"].get()
        try:
            _validate_timestamp_configuration(timestamp_column_name=npm_timestamp_column_name, time_unit=ts_unit)
        except ValueError as error:
            messagebox.showerror("NPM timestamp configuration incomplete", str(error))
            raise
        ts_units.append(ts_unit)
        npm_timestamp_column_names.append(npm_timestamp_column_name)
    return ts_units, npm_timestamp_column_names
