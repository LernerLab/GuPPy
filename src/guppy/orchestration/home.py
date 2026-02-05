import json
import logging
import os
import subprocess
import sys
from threading import Thread

import panel as pn

from .save_parameters import save_parameters
from .storenames import orchestrate_storenames_page
from ..frontend.input_parameters import ParameterForm
from ..frontend.path_selection import get_folder_path
from ..frontend.progress import readPBIncrementValues
from ..frontend.sidebar import Sidebar
from ..visualizePlot import visualizeResults

logger = logging.getLogger(__name__)


def readRawData(parameter_form):
    inputParameters = parameter_form.getInputParameters()
    subprocess.call([sys.executable, "-m", "guppy.orchestration.read_raw_data", json.dumps(inputParameters)])


def preprocess(parameter_form):
    inputParameters = parameter_form.getInputParameters()
    subprocess.call([sys.executable, "-m", "guppy.orchestration.preprocess", json.dumps(inputParameters)])


def psthComputation(parameter_form, current_dir):
    inputParameters = parameter_form.getInputParameters()
    inputParameters["curr_dir"] = current_dir
    subprocess.call([sys.executable, "-m", "guppy.computePsth", json.dumps(inputParameters)])


def build_homepage():
    pn.extension()
    global folder_path
    folder_path = get_folder_path()
    current_dir = os.getcwd()

    template = pn.template.BootstrapTemplate(title="Input Parameters GUI")
    parameter_form = ParameterForm(folder_path=folder_path, template=template)
    sidebar = Sidebar(template=template)

    # ------------------------------------------------------------------------------------------------------------------
    # onclick closure functions for sidebar buttons
    def onclickProcess(event=None):
        inputParameters = parameter_form.getInputParameters()
        save_parameters(inputParameters=inputParameters)

    def onclickStorenames(event=None):
        inputParameters = parameter_form.getInputParameters()
        orchestrate_storenames_page(inputParameters)

    def onclickVisualization(event=None):
        inputParameters = parameter_form.getInputParameters()
        visualizeResults(inputParameters)

    def onclickreaddata(event=None):
        thread = Thread(target=readRawData, args=(parameter_form,))
        thread.start()
        readPBIncrementValues(sidebar.read_progress)
        thread.join()

    def onclickpreprocess(event=None):
        thread = Thread(target=preprocess, args=(parameter_form,))
        thread.start()
        readPBIncrementValues(sidebar.extract_progress)
        thread.join()

    def onclickpsth(event=None):
        thread = Thread(target=psthComputation, args=(parameter_form, current_dir))
        thread.start()
        readPBIncrementValues(sidebar.psth_progress)
        thread.join()

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
        "getInputParameters": parameter_form.getInputParameters,
    }
    template._widgets = {
        "files_1": parameter_form.files_1,
    }

    return template
