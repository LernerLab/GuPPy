import glob
import json
import logging
import os
import socket
from pathlib import Path
from random import randint

import numpy as np
import panel as pn

from .storenames_instructions import StorenamesInstructions, StorenamesInstructionsNPM
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


def _save(d, select_location, storenames_selector):
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
    if not os.path.exists(select_location):
        os.mkdir(select_location)

    np.savetxt(os.path.join(select_location, "storesList.csv"), arr, delimiter=",", fmt="%s")
    logger.info(f"Storeslist file saved at {select_location}")
    logger.info("Storeslist : \n" + str(arr))


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
        storenames_instructions = StorenamesInstructionsNPM(folder_path=folder_path)
    else:
        storenames_instructions = StorenamesInstructions(folder_path=folder_path)
    storenames_selector = StorenamesSelector(allnames=allnames)

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
        global storenames
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

    # on clicking 'Select Storenames' button, following function is executed
    def update_values(event):
        global storenames, vars_list

        arr = storenames_selector.get_take_widgets()
        new_arr = []
        for i in range(len(arr[1])):
            for j in range(arr[1][i]):
                new_arr.append(arr[0][i])
        if len(new_arr) > 0:
            storenames = storenames_selector.get_cross_selector() + new_arr
        else:
            storenames = storenames_selector.get_cross_selector()
        storenames_selector.set_change_widgets(storenames)

        storenames_cache = dict()
        if os.path.exists(os.path.join(Path.home(), ".storesList.json")):
            with open(os.path.join(Path.home(), ".storesList.json")) as f:
                storenames_cache = json.load(f)

        storenames_selector.configure_storenames(
            storename_dropdowns=storename_dropdowns,
            storename_textboxes=storename_textboxes,
            storenames=storenames,
            storenames_cache=storenames_cache,
        )

    # on clicking save button, following function is executed
    def save_button(event=None):
        global storenames
        d = storenames_selector.get_literal_input_2()
        select_location = storenames_selector.get_select_location()
        _save(d=d, select_location=select_location, storenames_selector=storenames_selector)
        storenames_selector.set_path(os.path.join(select_location, "storesList.csv"))

    # ------------------------------------------------------------------------------------------------------------------

    # Connect button callbacks
    button_name_to_onclick_fn = {
        "update_options": update_values,
        "save": save_button,
        "overwrite_button": overwrite_button_actions,
        "show_config_button": fetchValues,
    }
    storenames_selector.attach_callbacks(button_name_to_onclick_fn)

    template.main.append(pn.Row(storenames_instructions.widget, storenames_selector.widget))

    # creating widgets, adding them to template and showing a GUI on a new browser window
    number = scanPortsAndFind(start_port=5000, end_port=5200)
    template.show(port=number)
