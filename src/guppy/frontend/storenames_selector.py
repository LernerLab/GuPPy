import json
import logging

import panel as pn

# hv.extension()
pn.extension()

logger = logging.getLogger(__name__)


class StorenamesSelector:

    def __init__(self, allnames):
        self.alert = pn.pane.Alert("#### No alerts !!", alert_type="danger", height=80, width=600)
        if len(allnames) == 0:
            self.alert.object = (
                "####Alert !! \n No storenames found. There are not any TDT files or csv files to look for storenames."
            )

        # creating different buttons and selectors for the GUI
        self.cross_selector = pn.widgets.CrossSelector(
            name="Store Names Selection", value=[], options=allnames, width=600
        )
        self.multi_choice = pn.widgets.MultiChoice(
            name="Select Storenames which you want more than once (multi-choice: multiple options selection)",
            value=[],
            options=allnames,
        )

        self.literal_input_1 = pn.widgets.LiteralInput(
            name="Number of times you want the above storename (list)", value=[], type=list
        )
        # self.literal_input_2 = pn.widgets.LiteralInput(name='Names for Storenames (list)', type=list)

        self.repeat_storenames = pn.widgets.Checkbox(name="Storenames to repeat", value=False)
        self.repeat_storename_wd = pn.WidgetBox("", width=600)

        self.repeat_storenames.link(self.repeat_storename_wd, callbacks={"value": self.callback})
        # self.repeat_storename_wd = pn.WidgetBox('Storenames to repeat (leave blank if not needed)', multi_choice, literal_input_1, background="white", width=600)

        self.update_options = pn.widgets.Button(name="Select Storenames", width=600)
        self.save = pn.widgets.Button(name="Save", width=600)

        self.text = pn.widgets.LiteralInput(value=[], name="Selected Store Names", type=list, width=600)

        self.path = pn.widgets.TextInput(name="Location to Stores List file", width=600)

        self.mark_down_for_overwrite = pn.pane.Markdown(
            """ Select option from below if user wants to over-write a file or create a new file.
                        **Creating a new file will make a new output folder and will get saved at that location.**
                        If user selects to over-write a file **Select location of the file to over-write** will provide
                        the existing options of the output folders where user needs to over-write the file""",
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

        self.literal_input_2 = pn.widgets.CodeEditor(
            value="""{}""", theme="tomorrow", language="json", height=250, width=600
        )

        self.take_widgets = pn.WidgetBox(self.multi_choice, self.literal_input_1)

        self.change_widgets = pn.WidgetBox(self.text)

        # Panel-based storename configuration (replaces Tkinter dialog)
        self.storename_config_widgets = pn.Column(visible=False)
        self.show_config_button = pn.widgets.Button(name="Show Selected Configuration", width=600)

        self.widget_2 = pn.Column(
            self.repeat_storenames,
            self.repeat_storename_wd,
            pn.Spacer(height=20),
            self.cross_selector,
            self.update_options,
            self.storename_config_widgets,
            pn.Spacer(height=10),
            self.text,
            self.literal_input_2,
            self.alert,
            self.mark_down_for_overwrite,
            self.overwrite_button,
            self.select_location,
            self.save,
            self.path,
        )

    def callback(self, target, event):
        if event.new == True:
            target.objects = [self.multi_choice, self.literal_input_1]
        elif event.new == False:
            target.clear()

    def get_select_location(self):
        return self.select_location.value

    def set_select_location_options(self, options):
        self.select_location.options = options

    def set_alert_message(self, message):
        self.alert.object = message

    def get_literal_input_2(self):  # TODO: come up with a better name for this method.
        d = json.loads(self.literal_input_2.value)
        return d

    def set_literal_input_2(self, d):  # TODO: come up with a better name for this method.
        self.literal_input_2.value = str(json.dumps(d, indent=2))

    def get_take_widgets(self):
        return [w.value for w in self.take_widgets]

    def set_change_widgets(self, value):
        for w in self.change_widgets:
            w.value = value

    def get_cross_selector(self):
        return self.cross_selector.value

    def set_path(self, value):
        self.path.value = value

    def attach_callbacks(self, button_name_to_onclick_fn: dict):
        for button_name, onclick_fn in button_name_to_onclick_fn.items():
            button = getattr(self, button_name)
            button.on_click(onclick_fn)
