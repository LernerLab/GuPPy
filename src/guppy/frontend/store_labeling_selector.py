import json
import logging

import panel as pn

from .store_labeling_config import StoreLabelingConfig

pn.extension()

logger = logging.getLogger(__name__)


class StoreLabelingSelector:
    """Panel widget for selecting, naming, and saving store_ids for a session.

    Parameters
    ----------
    allnames : list of str
        All store_ids discovered from the data files, offered as selectable
        options in the cross-selector and multi-choice widgets.
    """

    def __init__(self, allnames: list[str]) -> None:
        self.alert = pn.pane.Alert("#### No alerts !!", alert_type="danger", height=80, width=600)
        if len(allnames) == 0:
            self.alert.object = (
                "####Alert !! \n No store_ids found. There are not any TDT files or csv files to look for store_ids."
            )

        # creating different buttons and selectors for the GUI
        self.cross_selector = pn.widgets.CrossSelector(name="Store Selection", value=[], options=allnames, width=600)
        self.multi_choice = pn.widgets.MultiChoice(
            name="Select Stores which you want more than once (multi-choice: multiple options selection)",
            value=[],
            options=allnames,
        )

        self.literal_input_1 = pn.widgets.LiteralInput(
            name="Number of times you want the above store (list)", value=[], type=list
        )

        self.repeat_stores = pn.widgets.Checkbox(name="Stores to repeat", value=False)
        self.repeat_store_wd = pn.WidgetBox("", width=600)

        self.repeat_stores.link(self.repeat_store_wd, callbacks={"value": self.callback})

        self.update_options = pn.widgets.Button(name="Select Stores", width=600)
        self.save = pn.widgets.Button(name="Save", width=600)

        self.text = pn.widgets.LiteralInput(value=[], name="Selected Stores", type=list, width=600)

        self.path = pn.widgets.TextInput(name="Location to storesList file", width=600)

        self.mark_down_for_overwrite = pn.pane.Markdown(
            """
**Choose how to save this store_array:**

- **create_new_file** — create a new output folder. Optionally set **Run name** below; leave blank to use the next available integer.
- **over_write_file** — replace an existing output folder. Pick which one in **Select location of the file to over-write**.
            """,
            width=600,
        )

        self.run_name = pn.widgets.TextInput(
            name="Run name",
            value="",
            placeholder="optional — defaults to next available integer",
            width=600,
        )

        self.select_location = pn.widgets.Select(
            name="Select location of the file to over-write", value="None", options=["None"], width=600
        )

        self.overwrite_button = pn.widgets.MenuButton(
            name="over-write storeslist file or create a new one?  ",
            items=["over_write_file", "create_new_file"],
            button_type="default",
            split=True,
            width=600,
        )
        self._current_overwrite_mode = "create_new_file"

        self.literal_input_2 = pn.widgets.CodeEditor(
            value="""{}""", theme="tomorrow", language="json", height=250, width=600
        )

        self.take_widgets = pn.WidgetBox(self.multi_choice, self.literal_input_1)

        self.change_widgets = pn.WidgetBox(self.text)

        # Panel-based store_id configuration (replaces Tkinter dialog)
        self.store_id_config_widgets = pn.Column(visible=False)
        self.show_config_button = pn.widgets.Button(name="Show Selected Configuration", width=600)

        self.widget = pn.Column(
            self.repeat_stores,
            self.repeat_store_wd,
            pn.Spacer(height=20),
            self.cross_selector,
            self.update_options,
            self.store_id_config_widgets,
            pn.Spacer(height=10),
            self.text,
            self.literal_input_2,
            self.alert,
            self.mark_down_for_overwrite,
            self.overwrite_button,
            self.run_name,
            self.select_location,
            self.save,
            self.path,
        )

    def callback(self, target: pn.WidgetBox, event: object) -> None:
        """Show or hide the store_ids-to-repeat widget box based on the checkbox state.

        Parameters
        ----------
        target : pn.WidgetBox
            The widget box to populate or clear.
        event : param.parameterized.Event
            The watch event fired by ``repeat_stores``; ``event.new`` is the
            new checkbox value.
        """
        if event.new == True:
            target.objects = [self.multi_choice, self.literal_input_1]
        elif event.new == False:
            target.clear()

    def get_select_location(self) -> str:
        """Return the currently selected overwrite-location option.

        Returns
        -------
        str
            Value of the ``select_location`` selector widget.
        """
        return self.select_location.value

    def set_select_location_options(self, options: list[str]) -> None:
        """Replace the options in the overwrite-location selector.

        Parameters
        ----------
        options : list of str
            New list of location options to display.
        """
        self.select_location.options = options

    def set_alert_message(self, message: str) -> None:
        """Set the text shown in the alert pane.

        Parameters
        ----------
        message : str
            Markdown-formatted alert message.
        """
        self.alert.object = message

    def get_literal_input_2(self) -> dict[str, object]:  # TODO: come up with a better name for this method.
        """Parse and return the JSON store_ids mapping from the code editor widget.

        Returns
        -------
        dict
            Parsed JSON object from the ``literal_input_2`` code editor.
        """
        store_labeling_config = json.loads(self.literal_input_2.value)
        return store_labeling_config

    def set_literal_input_2(
        self, store_labeling_config: dict[str, object]
    ) -> None:  # TODO: come up with a better name for this method.
        """Serialise ``store_labeling_config`` as pretty-printed JSON and set the code editor value.

        Parameters
        ----------
        store_labeling_config : dict
            Dictionary to serialise into the ``literal_input_2`` code editor.
        """
        self.literal_input_2.value = str(json.dumps(store_labeling_config, indent=2))

    def get_take_widgets(self) -> list[object]:
        """Return the current values of the repeat-store_ids widgets.

        Returns
        -------
        list
            One entry per widget in ``take_widgets`` containing that widget's
            current value.
        """
        return [widget.value for widget in self.take_widgets]

    def set_change_widgets(self, value: object) -> None:
        """Set all ``change_widgets`` to ``value``.

        Parameters
        ----------
        value : object
            Value to assign to every widget in the ``change_widgets`` box.
        """
        for widget in self.change_widgets:
            widget.value = value

    def get_cross_selector(self) -> list[str]:
        """Return the store_ids currently selected in the cross-selector.

        Returns
        -------
        list of str
            Values selected by the user in ``cross_selector``.
        """
        return self.cross_selector.value

    def set_path(self, value: str) -> None:
        """Set the displayed path in the location text input.

        Parameters
        ----------
        value : str
            Path string to display in the ``path`` widget.
        """
        self.path.value = value

    def attach_callbacks(self, button_name_to_onclick_fn: dict[str, object]) -> None:
        """Register click-handler callbacks on selector buttons.

        Parameters
        ----------
        button_name_to_onclick_fn : dict
            Mapping from button attribute name (e.g. ``"save"``) to the callable
            that should be invoked when that button is clicked.
        """
        for button_name, onclick_fn in button_name_to_onclick_fn.items():
            button = getattr(self, button_name)
            if button_name == "overwrite_button":
                # Wrap the user callback so we can also remember the current
                # mode (for get_overwrite_mode) and hide the run-name field in
                # overwrite mode where it has no effect.
                def remember_then_call(event: object, _user_callback: object = onclick_fn) -> None:
                    self._current_overwrite_mode = event.new
                    self.run_name.visible = event.new == "create_new_file"
                    _user_callback(event)

                button.on_click(remember_then_call)
            else:
                button.on_click(onclick_fn)

    def attach_run_name_watcher(self, callback: object) -> None:
        """Attach a watcher that fires when the run-name TextInput value changes.

        Parameters
        ----------
        callback : callable
            Function with signature ``callback(event)`` where ``event.new`` is
            the new run-name string.
        """
        self.run_name.param.watch(callback, "value")

    def get_run_name(self) -> str:
        """Return the current value of the run-name TextInput.

        Returns
        -------
        str
            Run-name string entered by the user (may be empty).
        """
        return self.run_name.value

    def get_overwrite_mode(self) -> str:
        """Return the current overwrite-vs-create mode.

        Returns
        -------
        str
            ``"over_write_file"`` or ``"create_new_file"``.
        """
        return self._current_overwrite_mode

    def configure_store_ids(
        self,
        store_id_dropdowns: dict[str, pn.widgets.Select],
        store_id_textboxes: dict[str, pn.widgets.TextInput],
        store_ids: list[str],
        store_id_to_store_labels: dict[str, list[str]],
    ) -> None:
        """Build the store_id-configuration panel and make it visible.

        Parameters
        ----------
        store_id_dropdowns : dict
            Mutable mapping populated by ``StoreLabelingConfig`` with dropdown widgets.
        store_id_textboxes : dict
            Mutable mapping populated by ``StoreLabelingConfig`` with text-input widgets.
        store_ids : list of str
            Raw store_ids selected by the user.
        store_id_to_store_labels : dict
            Previously saved store_id assignments for pre-population.
        """
        # Create Panel widgets for store_id configuration
        self.store_labeling_config = StoreLabelingConfig(
            show_config_button=self.show_config_button,
            store_id_dropdowns=store_id_dropdowns,
            store_id_textboxes=store_id_textboxes,
            store_ids=store_ids,
            store_id_to_store_labels=store_id_to_store_labels,
        )

        # Update the configuration panel
        self.store_id_config_widgets.objects = self.store_labeling_config.config_widgets
        self.store_id_config_widgets.visible = len(store_ids) > 0
