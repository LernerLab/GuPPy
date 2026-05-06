import json
import logging
import os
import subprocess
import sys
from threading import Thread

import panel as pn

from .save_parameters import save_parameters
from .storenames import orchestrate_storenames_page
from .visualize import visualizeResults
from ..frontend.input_parameters import ParameterForm
from ..frontend.progress import PB_STEPS_FILE, readPBIncrementValues
from ..frontend.sidebar import Sidebar

logger = logging.getLogger(__name__)


def readRawData(inputParameters):
    """
    Launch the raw-data extraction step in a subprocess.

    Parameters
    ----------
    inputParameters : dict
        Pipeline input parameters serialized to JSON for the subprocess.
    """
    subprocess.call([sys.executable, "-m", "guppy.orchestration.read_raw_data", json.dumps(inputParameters)])


def preprocess(inputParameters):
    """
    Launch the preprocessing step (timestamp correction, z-score) in a subprocess.

    Parameters
    ----------
    inputParameters : dict
        Pipeline input parameters serialized to JSON for the subprocess.
    """
    subprocess.call([sys.executable, "-m", "guppy.orchestration.preprocess", json.dumps(inputParameters)])


def psthComputation(inputParameters):
    """
    Launch the PSTH computation step in a subprocess.

    Parameters
    ----------
    inputParameters : dict
        Pipeline input parameters serialized to JSON for the subprocess.
    """
    subprocess.call([sys.executable, "-m", "guppy.orchestration.psth", json.dumps(inputParameters)])


def build_homepage(*, start_path=None):
    """
    Build and return the GuPPy Panel web-application template.

    Parameters
    ----------
    start_path : str or None, optional
        Initial directory shown in the folder-selection widget.  When None the
        widget starts in the current working directory.

    Returns
    -------
    template : pn.template.BootstrapTemplate
        Fully wired Panel template ready to be served or shown.
    """
    pn.extension(notifications=True)
    current_dir = os.getcwd()

    template = pn.template.BootstrapTemplate(title="Input Parameters GUI")
    parameter_form = ParameterForm(template=template, start_path=start_path)
    sidebar = Sidebar(template=template)

    # ------------------------------------------------------------------------------------------------------------------
    # onclick closure functions for sidebar buttons
    def _getInputParametersOrNotify():
        try:
            return parameter_form.getInputParameters()
        except Exception as e:
            pn.state.notifications.error(str(e), duration=0)
            return None

    def onclickProcess(event=None):
        inputParameters = _getInputParametersOrNotify()
        if inputParameters is None:
            return
        save_parameters(inputParameters=inputParameters)

    def onclickStorenames(event=None):
        inputParameters = _getInputParametersOrNotify()
        if inputParameters is None:
            return
        orchestrate_storenames_page(inputParameters)

    def onclickVisualization(event=None):
        inputParameters = _getInputParametersOrNotify()
        if inputParameters is None:
            return
        try:
            visualizeResults(inputParameters)
        except ValueError as e:
            pn.state.notifications.error(str(e), duration=0)

    def onclickreaddata(event=None):
        inputParameters = _getInputParametersOrNotify()
        if inputParameters is None:
            return
        thread = Thread(target=readRawData, args=(inputParameters,))
        thread.start()
        error_msg = readPBIncrementValues(sidebar.read_progress, file_path=PB_STEPS_FILE)
        thread.join()
        if error_msg:
            pn.state.notifications.error(error_msg, duration=0)

    def onclickpreprocess(event=None):
        inputParameters = _getInputParametersOrNotify()
        if inputParameters is None:
            return
        thread = Thread(target=preprocess, args=(inputParameters,))
        thread.start()
        error_msg = readPBIncrementValues(sidebar.extract_progress, file_path=PB_STEPS_FILE)
        thread.join()
        if error_msg:
            pn.state.notifications.error(error_msg, duration=0)

    def onclickpsth(event=None):
        inputParameters = _getInputParametersOrNotify()
        if inputParameters is None:
            return
        inputParameters["curr_dir"] = current_dir
        thread = Thread(target=psthComputation, args=(inputParameters,))
        thread.start()
        error_msg = readPBIncrementValues(sidebar.psth_progress, file_path=PB_STEPS_FILE)
        thread.join()
        if error_msg:
            pn.state.notifications.error(error_msg, duration=0)

    # ------------------------------------------------------------------------------------------------------------------

    button_name_to_onclick_fn = {
        "save_button": onclickProcess,
        "open_storenames": onclickStorenames,
        "read_rawData": onclickreaddata,
        "preprocess": onclickpreprocess,
        "psth_computation": onclickpsth,
        "open_visualization": onclickVisualization,
    }
    sidebar.attach_callbacks(button_name_to_onclick_fn=button_name_to_onclick_fn)
    sidebar.add_to_template()

    # Expose minimal hooks and widgets to enable programmatic testing
    template._hooks = {
        "onclickProcess": onclickProcess,
        "onclickVisualization": onclickVisualization,
        "getInputParameters": parameter_form.getInputParameters,
    }
    template._widgets = {
        "files_1": parameter_form.files_1,
        "source_mode": parameter_form.source_mode,
        "dandi_selector": parameter_form.dandi_selector,
    }

    return template
