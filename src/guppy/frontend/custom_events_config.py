import logging

# holoviews must be imported before the first pn.extension() call so Panel wires up the HoloViews
# bokeh opts namespace that the visualization step relies on; importing this module first (it sorts
# ahead of store_ids in home.py) would otherwise leave those opts unregistered. Mirrors the
# import-then-extension ordering in store_labeling_instructions.py.
import holoviews as hv  # noqa: F401
import panel as pn

from .row_controls import column_heading, delete_row_button

pn.extension()

logger = logging.getLogger(__name__)

# The name column's width, shared by its heading and its rows; the timestamps column takes the rest.
_NAME_WIDTH = 220

_INSTRUCTIONS = """
## Import Custom Events

Add behavioral events you derived outside GuPPy, such as movement onsets or
events scored by hand. For each event, enter its name and paste its timestamps
in seconds, one per row, as copied from a spreadsheet column. Click
**Add Event** for another event, then **Save**.

Each event is saved in this session as ``<name>.csv`` and is offered as a store
on the Label Stores page. This step is optional: with no custom events, leave
the table empty and go on to Label Stores.
"""


class CustomEventsConfig:
    """Panel widget for pasting one or more named custom events for a session.

    Renders a header, a table of event rows (each with a name field, a paste box
    for timestamps and a delete button), an "Add Event" button, an "Overwrite
    existing" checkbox, a "Save" button, and an alert pane shown once there is a
    message. The widget only collects user input and displays alerts; writing the
    CSV files is the orchestration layer's responsibility.
    """

    def __init__(self) -> None:
        self.rows: list[tuple[pn.widgets.TextInput, pn.widgets.TextAreaInput]] = []

        self.instructions = pn.pane.Markdown(_INSTRUCTIONS, sizing_mode="stretch_width")
        self.event_rows = pn.Column(sizing_mode="stretch_width")
        self.add_event = pn.widgets.Button(name="Add Event", button_type="default", icon="plus")
        self.overwrite = pn.widgets.Checkbox(name="Overwrite existing events with the same name", value=False)
        self.save = pn.widgets.Button(name="Save", button_type="primary")
        self.alert = pn.pane.Alert("", alert_type="primary", sizing_mode="stretch_width", visible=False)

        self.add_event_row()

        self.widget = pn.Column(
            self.instructions,
            pn.Row(
                column_heading(text="Event Name", width=_NAME_WIDTH),
                column_heading(text="Timestamps (s)"),
                sizing_mode="stretch_width",
                margin=(0, 0, 2, 0),
            ),
            self.event_rows,
            self.add_event,
            self.overwrite,
            self.save,
            self.alert,
            sizing_mode="stretch_width",
        )

    def add_event_row(self, event: object = None) -> None:
        """Append a fresh, empty event row (name field + paste box).

        Parameters
        ----------
        event : object, optional
            Unused click-event argument so this can be wired directly to the
            "Add Event" button.
        """
        name_input = pn.widgets.TextInput(placeholder="e.g. movement_onset", width=_NAME_WIDTH)
        timestamps_input = pn.widgets.TextAreaInput(
            placeholder="0.0\n1.5\n2.3", height=120, sizing_mode="stretch_width"
        )
        remove = delete_row_button(description="Remove this event")
        row = pn.Row(name_input, timestamps_input, remove, sizing_mode="stretch_width", margin=(0, 0, 4, 0))
        entry = (name_input, timestamps_input)

        def remove_row(event: object) -> None:
            self.rows.remove(entry)
            self.event_rows.remove(row)

        remove.on_click(remove_row)
        self.rows.append(entry)
        self.event_rows.append(row)

    def get_rows(self) -> list[tuple[str, str]]:
        """Return the ``(name, pasted_text)`` pair currently entered in each row.

        Returns
        -------
        list of tuple of (str, str)
            One ``(name, pasted_text)`` pair per event row, in display order.
        """
        return [(name_input.value, timestamps_input.value) for name_input, timestamps_input in self.rows]

    def set_alert_message(self, message: str) -> None:
        """Set the text shown in the alert pane.

        Parameters
        ----------
        message : str
            Markdown-formatted alert message.
        """
        self.alert.object = message
        self.alert.visible = True

    def attach_callbacks(self, button_name_to_onclick_fn: dict[str, object]) -> None:
        """Register click-handler callbacks on the widget's buttons.

        Parameters
        ----------
        button_name_to_onclick_fn : dict
            Mapping from button attribute name (``"add_event"`` or ``"save"``) to
            the callable invoked when that button is clicked.
        """
        for button_name, onclick_fn in button_name_to_onclick_fn.items():
            button = getattr(self, button_name)
            button.on_click(onclick_fn)
