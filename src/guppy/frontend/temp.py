import glob
import json
import logging
import os
import socket
from pathlib import Path
from random import randint

import numpy as np
import panel as pn

from .npm_channel_selector import NpmChannelSelector
from .storenames_instructions import StorenamesInstructions
from .storenames_selector import StorenamesSelector

# hv.extension()
pn.extension()

logger = logging.getLogger(__name__)


def scanPortsAndFind(start_port=5000, end_port=5200, host="127.0.0.1"):
    while True:
        port = randint(start_port, end_port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.001)  # Set timeout to avoid long waiting on closed ports
        result = sock.connect_ex((host, port))
        if result == 0:  # If the connection is successful, the port is open
            continue
        else:
            break

    return port


def takeOnlyDirs(paths):
    removePaths = []
    for p in paths:
        if os.path.isfile(p):
            removePaths.append(p)
    return list(set(paths) - set(removePaths))


# function to show location for over-writing or creating a new stores list file.
def show_dir(filepath):
    i = 1
    while True:
        basename = os.path.basename(filepath)
        op = os.path.join(filepath, basename + "_output_" + str(i))
        if not os.path.exists(op):
            break
        i += 1
    return op


def make_dir(filepath):
    i = 1
    while True:
        basename = os.path.basename(filepath)
        op = os.path.join(filepath, basename + "_output_" + str(i))
        if not os.path.exists(op):
            os.mkdir(op)
            break
        i += 1

    return op


def _fetchValues(text, storenames, storename_dropdowns, storename_textboxes, d):
    if not storename_dropdowns or not len(storenames) > 0:
        return "####Alert !! \n No storenames selected."

    storenames_cache = dict()
    if os.path.exists(os.path.join(Path.home(), ".storesList.json")):
        with open(os.path.join(Path.home(), ".storesList.json")) as f:
            storenames_cache = json.load(f)

    comboBoxValues, textBoxValues = [], []
    dropdown_keys = list(storename_dropdowns.keys())
    textbox_keys = list(storename_textboxes.keys()) if storename_textboxes else []

    # Get dropdown values
    for key in dropdown_keys:
        comboBoxValues.append(storename_dropdowns[key].value)

    # Get textbox values (matching with dropdown keys)
    for key in dropdown_keys:
        if key in storename_textboxes:
            textbox_value = storename_textboxes[key].value or ""
            textBoxValues.append(textbox_value)

            # Validation: Check for whitespace
            if len(textbox_value.split()) > 1:
                return "####Alert !! \n Whitespace is not allowed in the text box entry."

            # Validation: Check for empty required fields
            dropdown_value = storename_dropdowns[key].value
            if (
                not textbox_value
                and dropdown_value not in storenames_cache
                and dropdown_value in ["control", "signal", "event TTLs"]
            ):
                return "####Alert !! \n One of the text box entry is empty."
        else:
            # For cached values, use the dropdown value directly
            textBoxValues.append(storename_dropdowns[key].value)

    if len(comboBoxValues) != len(textBoxValues):
        return "####Alert !! \n Number of entries in combo box and text box should be same."

    names_for_storenames = []
    for i in range(len(comboBoxValues)):
        if comboBoxValues[i] == "control" or comboBoxValues[i] == "signal":
            if "_" in textBoxValues[i]:
                return "####Alert !! \n Please do not use underscore in region name."
            names_for_storenames.append("{}_{}".format(comboBoxValues[i], textBoxValues[i]))
        elif comboBoxValues[i] == "event TTLs":
            names_for_storenames.append(textBoxValues[i])
        else:
            names_for_storenames.append(comboBoxValues[i])

    d["storenames"] = text.value
    d["names_for_storenames"] = names_for_storenames
    return "#### No alerts !!"


# function to show GUI and save
def saveStorenames(inputParameters, events, flags, folder_path):

    logger.debug("Saving stores list file.")
    # getting input parameters
    inputParameters = inputParameters

    # Headless path: if storenames_map provided, write storesList.csv without building the Panel UI
    storenames_map = inputParameters.get("storenames_map")
    if isinstance(storenames_map, dict) and len(storenames_map) > 0:
        op = make_dir(folder_path)
        arr = np.asarray([list(storenames_map.keys()), list(storenames_map.values())], dtype=str)
        np.savetxt(os.path.join(op, "storesList.csv"), arr, delimiter=",", fmt="%s")
        logger.info(f"Storeslist file saved at {op}")
        logger.info("Storeslist : \n" + str(arr))
        return

    # Get storenames from extractor's events property
    allnames = events

    # creating GUI template
    template = pn.template.BootstrapTemplate(title="Storenames GUI - {}".format(os.path.basename(folder_path)))

    if "data_np_v2" in flags or "data_np" in flags or "event_np" in flags:
        npm_channel_selector = NpmChannelSelector(folder_path=folder_path)

    storenames_instructions = StorenamesInstructions()
    storenames_selector = StorenamesSelector(allnames=allnames)
    alert = storenames_selector.alert

    storenames = []
    storename_dropdowns = {}
    storename_textboxes = {}

    # ------------------------------------------------------------------------------------------------------------------
    # onclick closure functions
    # on clicking overwrite_button, following function is executed
    def overwrite_button_actions(event):
        if event.new == "over_write_file":
            options = takeOnlyDirs(glob.glob(os.path.join(folder_path, "*_output_*")))
            storenames_selector.set_select_location_options(options=options)
        else:
            options = [show_dir(folder_path)]
            storenames_selector.set_select_location_options(options=options)

    def fetchValues(event):
        d = dict()
        alert_message = _fetchValues(
            text=storenames_selector.text,
            storenames=storenames,
            storename_dropdowns=storename_dropdowns,
            storename_textboxes=storename_textboxes,
            d=d,
        )
        storenames_selector.set_alert_message(alert_message)
        storenames_selector.set_literal_input_2(d=d)
        global storenames

    # Panel-based storename configuration (replaces Tkinter dialog)
    storename_config_widgets = pn.Column(visible=False)
    show_config_button = pn.widgets.Button(name="Show Selected Configuration", width=600)

    # on clicking 'Select Storenames' button, following function is executed
    def update_values(event):
        global storenames, vars_list
        arr = []
        for w in take_widgets:
            arr.append(w.value)

        new_arr = []

        for i in range(len(arr[1])):
            for j in range(arr[1][i]):
                new_arr.append(arr[0][i])

        if len(new_arr) > 0:
            storenames = cross_selector.value + new_arr
        else:
            storenames = cross_selector.value

        for w in change_widgets:
            w.value = storenames

        storenames_cache = dict()
        if os.path.exists(os.path.join(Path.home(), ".storesList.json")):
            with open(os.path.join(Path.home(), ".storesList.json")) as f:
                storenames_cache = json.load(f)

        # Create Panel widgets for storename configuration
        config_widgets = []
        storename_dropdowns.clear()
        storename_textboxes.clear()

        if len(storenames) > 0:
            config_widgets.append(
                pn.pane.Markdown(
                    "## Configure Storenames\nSelect appropriate options for each storename and provide names as needed:"
                )
            )

            for i, storename in enumerate(storenames):
                # Create a row for each storename
                row_widgets = []

                # Label
                label = pn.pane.Markdown(f"**{storename}:**")
                row_widgets.append(label)

                # Dropdown options
                if storename in storenames_cache:
                    options = storenames_cache[storename]
                    default_value = options[0] if options else ""
                else:
                    options = ["", "control", "signal", "event TTLs"]
                    default_value = ""

                # Create unique key for widget
                widget_key = (
                    f"{storename}_{i}"
                    if f"{storename}_{i}" not in storename_dropdowns
                    else f"{storename}_{i}_{len(storename_dropdowns)}"
                )

                dropdown = pn.widgets.Select(name="Type", value=default_value, options=options, width=150)
                storename_dropdowns[widget_key] = dropdown
                row_widgets.append(dropdown)

                # Text input (only show if not cached or if control/signal/event TTLs selected)
                if storename not in storenames_cache or default_value in ["control", "signal", "event TTLs"]:
                    textbox = pn.widgets.TextInput(
                        name="Name", value="", placeholder="Enter region/event name", width=200
                    )
                    storename_textboxes[widget_key] = textbox
                    row_widgets.append(textbox)

                    # Add helper text based on selection
                    def create_help_function(dropdown_widget, help_pane_container):
                        @pn.depends(dropdown_widget.param.value, watch=True)
                        def update_help(dropdown_value):
                            if dropdown_value == "control":
                                help_pane_container[0] = pn.pane.Markdown(
                                    "*Type appropriate region name*", styles={"color": "gray", "font-size": "12px"}
                                )
                            elif dropdown_value == "signal":
                                help_pane_container[0] = pn.pane.Markdown(
                                    "*Type appropriate region name*", styles={"color": "gray", "font-size": "12px"}
                                )
                            elif dropdown_value == "event TTLs":
                                help_pane_container[0] = pn.pane.Markdown(
                                    "*Type event name for the TTLs*", styles={"color": "gray", "font-size": "12px"}
                                )
                            else:
                                help_pane_container[0] = pn.pane.Markdown(
                                    "", styles={"color": "gray", "font-size": "12px"}
                                )

                        return update_help

                    help_container = [pn.pane.Markdown("")]
                    help_function = create_help_function(dropdown, help_container)
                    help_function(dropdown.value)  # Initialize
                    row_widgets.append(help_container[0])

                # Add the row to config widgets
                config_widgets.append(pn.Row(*row_widgets, margin=(5, 0)))

            # Add show button
            config_widgets.append(pn.Spacer(height=20))
            config_widgets.append(show_config_button)
            config_widgets.append(
                pn.pane.Markdown(
                    "*Click 'Show Selected Configuration' to apply your selections.*",
                    styles={"font-size": "12px", "color": "gray"},
                )
            )

        # Update the configuration panel
        storename_config_widgets.objects = config_widgets
        storename_config_widgets.visible = len(storenames) > 0

    # on clicking save button, following function is executed
    def save_button(event=None):
        global storenames

        d = json.loads(literal_input_2.value)
        arr1, arr2 = np.asarray(d["storenames"]), np.asarray(d["names_for_storenames"])

        if np.where(arr2 == "")[0].size > 0:
            storenames_selector.set_alert_message("#### Alert !! \n Empty string in the list names_for_storenames.")
            logger.error("Empty string in the list names_for_storenames.")
            raise Exception("Empty string in the list names_for_storenames.")
        else:
            storenames_selector.set_alert_message("#### No alerts !!")

        if arr1.shape[0] != arr2.shape[0]:
            storenames_selector.set_alert_message(
                "#### Alert !! \n Length of list storenames and names_for_storenames is not equal."
            )
            logger.error("Length of list storenames and names_for_storenames is not equal.")
            raise Exception("Length of list storenames and names_for_storenames is not equal.")
        else:
            storenames_selector.set_alert_message("#### No alerts !!")

        if not os.path.exists(os.path.join(Path.home(), ".storesList.json")):
            storenames_cache = dict()

            for i in range(arr1.shape[0]):
                if arr1[i] in storenames_cache:
                    storenames_cache[arr1[i]].append(arr2[i])
                    storenames_cache[arr1[i]] = list(set(storenames_cache[arr1[i]]))
                else:
                    storenames_cache[arr1[i]] = [arr2[i]]

            with open(os.path.join(Path.home(), ".storesList.json"), "w") as f:
                json.dump(storenames_cache, f, indent=4)
        else:
            with open(os.path.join(Path.home(), ".storesList.json")) as f:
                storenames_cache = json.load(f)

            for i in range(arr1.shape[0]):
                if arr1[i] in storenames_cache:
                    storenames_cache[arr1[i]].append(arr2[i])
                    storenames_cache[arr1[i]] = list(set(storenames_cache[arr1[i]]))
                else:
                    storenames_cache[arr1[i]] = [arr2[i]]

            with open(os.path.join(Path.home(), ".storesList.json"), "w") as f:
                json.dump(storenames_cache, f, indent=4)

        arr = np.asarray([arr1, arr2])
        logger.info(arr)
        if not os.path.exists(select_location.value):
            os.mkdir(select_location.value)

        np.savetxt(os.path.join(select_location.value, "storesList.csv"), arr, delimiter=",", fmt="%s")
        path.value = os.path.join(select_location.value, "storesList.csv")
        logger.info(f"Storeslist file saved at {select_location.value}")
        logger.info("Storeslist : \n" + str(arr))

    # Connect button callbacks
    update_options.on_click(update_values)
    show_config_button.on_click(fetchValues)
    save.on_click(save_button)
    overwrite_button.on_click(overwrite_button_actions)

    # creating widgets, adding them to template and showing a GUI on a new browser window
    number = scanPortsAndFind(start_port=5000, end_port=5200)

    if "data_np_v2" in flags or "data_np" in flags or "event_np" in flags:
        widget_1 = pn.Column(
            "# " + os.path.basename(folder_path),
            storenames_instructions.mark_down,
            npm_channel_selector.mark_down_np,
            npm_channel_selector.plot_select,
            npm_channel_selector.plot_pane,
        )
        widget_2 = pn.Column(
            repeat_storenames,
            repeat_storename_wd,
            pn.Spacer(height=20),
            cross_selector,
            update_options,
            storename_config_widgets,
            pn.Spacer(height=10),
            text,
            literal_input_2,
            alert,
            mark_down_for_overwrite,
            overwrite_button,
            select_location,
            save,
            path,
        )
        template.main.append(pn.Row(widget_1, widget_2))

    else:
        widget_1 = pn.Column("# " + os.path.basename(folder_path), storenames_instructions.mark_down)
        widget_2 = pn.Column(
            repeat_storenames,
            repeat_storename_wd,
            pn.Spacer(height=20),
            cross_selector,
            update_options,
            storename_config_widgets,
            pn.Spacer(height=10),
            text,
            literal_input_2,
            alert,
            mark_down_for_overwrite,
            overwrite_button,
            select_location,
            save,
            path,
        )
        template.main.append(pn.Row(widget_1, widget_2))

    template.show(port=number)
