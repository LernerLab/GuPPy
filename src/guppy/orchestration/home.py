import json
import logging
import os
import subprocess
import sys
from threading import Thread
from typing import Callable

import panel as pn

from .import_custom_events import orchestrate_custom_events_page
from .storenames import orchestrate_storenames_page
from .visualize import visualizeResults
from ..frontend.input_parameters import ParameterForm
from ..frontend.progress import PB_STEPS_FILE, readPBIncrementValues
from ..frontend.sidebar import Sidebar

logger = logging.getLogger(__name__)


def readRawData(inputParameters: dict[str, object]) -> None:
    """
    Launch the raw-data extraction step in a subprocess.

    Parameters
    ----------
    inputParameters : dict
        Pipeline input parameters serialized to JSON for the subprocess.
    """
    subprocess.call([sys.executable, "-m", "guppy.orchestration.read_raw_data", json.dumps(inputParameters)])


def preprocess(inputParameters: dict[str, object]) -> None:
    """
    Launch the preprocessing step (timestamp correction, z-score) in a subprocess.

    Parameters
    ----------
    inputParameters : dict
        Pipeline input parameters serialized to JSON for the subprocess.
    """
    subprocess.call([sys.executable, "-m", "guppy.orchestration.preprocess", json.dumps(inputParameters)])


def psthComputation(inputParameters: dict[str, object]) -> None:
    """
    Launch the PSTH computation step in a subprocess.

    Parameters
    ----------
    inputParameters : dict
        Pipeline input parameters serialized to JSON for the subprocess.
    """
    subprocess.call([sys.executable, "-m", "guppy.orchestration.psth", json.dumps(inputParameters)])


def build_homepage(*, start_path: str | None = None) -> pn.template.BootstrapTemplate:
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
    def _getInputParametersOrNotify(*, require_selected_outputs: bool = False) -> dict[str, object] | None:
        try:
            input_parameters = parameter_form.getInputParameters()
            if require_selected_outputs:
                parameter_form.validate_selected_outputs_for_consumers()
            return input_parameters
        except Exception as e:
            pn.state.notifications.error(str(e), duration=0)
            return None

    def _run_worker_with_progress(
        worker: Callable[[dict[str, object]], None],
        progress_widget: pn.indicators.Progress,
        *,
        add_curr_dir: bool = False,
    ) -> None:
        # Shared launch pattern for the pipeline steps that run a worker in a background
        # thread while a progress bar polls PB_STEPS_FILE.
        inputParameters = _getInputParametersOrNotify(require_selected_outputs=True)
        if inputParameters is None:
            return
        if add_curr_dir:
            inputParameters["curr_dir"] = current_dir
        thread = Thread(target=worker, args=(inputParameters,))
        thread.start()
        error_message = readPBIncrementValues(progress_widget, file_path=PB_STEPS_FILE)
        thread.join()
        if error_message:
            pn.state.notifications.error(error_message, duration=0)

    def onclickImportCustomEvents(event: object = None) -> None:
        inputParameters = _getInputParametersOrNotify()
        if inputParameters is None:
            return
        orchestrate_custom_events_page(inputParameters)

    def onclickStorenames(event: object = None) -> None:
        inputParameters = _getInputParametersOrNotify()
        if inputParameters is None:
            return
        orchestrate_storenames_page(inputParameters)
        # Newly-created output dirs become available for filtering on the next
        # step without requiring the user to deselect/reselect their session.
        parameter_form.refresh_individual_outputs()
        parameter_form.refresh_group_outputs()

    def onclickVisualization(event: object = None) -> None:
        inputParameters = _getInputParametersOrNotify(require_selected_outputs=True)
        if inputParameters is None:
            return
        try:
            visualizeResults(inputParameters)
        except ValueError as e:
            pn.state.notifications.error(str(e), duration=0)

    def onclickreaddata(event: object = None) -> None:
        _run_worker_with_progress(readRawData, sidebar.read_progress)

    def onclickpreprocess(event: object = None) -> None:
        _run_worker_with_progress(preprocess, sidebar.extract_progress)

    def onclickpsth(event: object = None) -> None:
        _run_worker_with_progress(psthComputation, sidebar.psth_progress, add_curr_dir=True)

    # ------------------------------------------------------------------------------------------------------------------

    button_name_to_onclick_fn = {
        "import_custom_events": onclickImportCustomEvents,
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
        "onclickVisualization": onclickVisualization,
        "onclickImportCustomEvents": onclickImportCustomEvents,
        "onclickreaddata": onclickreaddata,
        "onclickpreprocess": onclickpreprocess,
        "onclickpsth": onclickpsth,
        "getInputParameters": parameter_form.getInputParameters,
    }
    template._widgets = {
        "files_1": parameter_form.files_1,
        "source_mode": parameter_form.source_mode,
        "dandi_selector": parameter_form.dandi_selector,
    }

    return template
