import logging
import os
from threading import Thread
from typing import Callable

import panel as pn

from .export_nwb import orchestrate_export_nwb_page
from .import_custom_events import orchestrate_custom_events_page
from .metadata import orchestrate_metadata_page
from .preprocess import run_preprocess_step
from .preprocess_view import open_preprocess_view
from .psth import run_psth_step
from .read_raw_data import run_read_raw_data_step
from .store_labeling import orchestrate_store_labeling_page
from .transients_view import open_transients_view
from .visualize import visualizeResults
from ..frontend.input_parameters import ParameterForm
from ..frontend.progress import PB_ERROR_FILE, PB_STEPS_FILE, poll_progress_step
from ..frontend.sidebar import Sidebar

logger = logging.getLogger(__name__)


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
    # Guards against launching a second pipeline step while one is still running — the
    # handlers no longer block the IOLoop, so without this a user could start overlapping
    # runs that clobber the shared progress file and output directories.
    step_running = {"active": False}

    template = pn.template.BootstrapTemplate(title="Input Parameters GUI")
    parameter_form = ParameterForm(template=template, start_path=start_path)
    sidebar = Sidebar(template=template)

    # ------------------------------------------------------------------------------------------------------------------
    # onclick closure functions for sidebar buttons
    def _getInputParametersOrNotify(*, require_selected_outputs: bool = False) -> dict[str, object] | None:
        try:
            input_parameters = parameter_form.getInputParameters()
            if require_selected_outputs:
                parameter_form.validate_selected_runs_for_consumers()
            return input_parameters
        except Exception as e:
            pn.state.notifications.error(str(e), duration=0)
            return None

    def _run_worker_with_progress(
        worker: Callable[[dict[str, object]], None],
        progress_widget: pn.indicators.Progress,
        *,
        add_curr_dir: bool = False,
        on_success: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        # Shared launch pattern for the pipeline steps that run a worker in a background
        # thread while a progress bar polls PB_STEPS_FILE.
        if step_running["active"]:
            pn.state.notifications.error("A pipeline step is already running; wait for it to finish.", duration=0)
            return
        inputParameters = _getInputParametersOrNotify(require_selected_outputs=True)
        if inputParameters is None:
            return
        if add_curr_dir:
            inputParameters["curr_dir"] = current_dir

        if os.path.exists(PB_STEPS_FILE):
            os.remove(PB_STEPS_FILE)
        progress_widget.value = 0
        progress_widget.bar_color = "success"

        step_running["active"] = True
        thread = Thread(target=worker, args=(inputParameters,))
        thread.start()

        # Poll progress from a periodic callback on the server IOLoop rather than
        # busy-looping and joining the worker here. A synchronous wait would block the
        # IOLoop for the whole step, so the main app would stop answering websocket
        # heartbeats and the browser tab would drop its connection and never recover.
        poller: dict[str, object] = {}

        def poll() -> None:
            done, error_message = poll_progress_step(
                progress_widget, file_path=PB_STEPS_FILE, error_file_path=PB_ERROR_FILE
            )
            if done or not thread.is_alive():
                poller["callback"].stop()
                step_running["active"] = False
                if error_message:
                    pn.state.notifications.error(error_message, duration=0)
                elif on_success is not None:
                    # e.g. open the step's result view once compute succeeded.
                    on_success(inputParameters)

        poller["callback"] = pn.state.add_periodic_callback(poll, period=100)

    def onclickImportCustomEvents(event: object = None) -> None:
        inputParameters = _getInputParametersOrNotify()
        if inputParameters is None:
            return
        orchestrate_custom_events_page(inputParameters)

    def onclickLabelStores(event: object = None) -> None:
        inputParameters = _getInputParametersOrNotify()
        if inputParameters is None:
            return
        orchestrate_store_labeling_page(inputParameters)
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
        _run_worker_with_progress(run_read_raw_data_step, sidebar.read_progress)

    def onclickpreprocess(event: object = None) -> None:
        def _open_view(inputParameters: dict[str, object]) -> None:
            open_preprocess_view(inputParameters["session_folders"], inputParameters)

        _run_worker_with_progress(run_preprocess_step, sidebar.extract_progress, on_success=_open_view)

    def onclickpsth(event: object = None) -> None:
        def _open_view(inputParameters: dict[str, object]) -> None:
            # Group averaging produces no per-session transient-peak plots, so no view.
            if not inputParameters.get("averageForGroup"):
                open_transients_view(inputParameters["session_folders"], inputParameters)

        _run_worker_with_progress(run_psth_step, sidebar.psth_progress, add_curr_dir=True, on_success=_open_view)

    def onclickMetadata(event: object = None) -> None:
        inputParameters = _getInputParametersOrNotify(require_selected_outputs=True)
        if inputParameters is None:
            return
        orchestrate_metadata_page(inputParameters)

    def onclickExportNwb(event: object = None) -> None:
        inputParameters = _getInputParametersOrNotify(require_selected_outputs=True)
        if inputParameters is None:
            return
        # Runs synchronously (like visualization) so the progress bar and
        # notifications update directly; per-session failures are reported and skipped.
        try:
            orchestrate_export_nwb_page(inputParameters, progress_bar=sidebar.export_progress)
        except ValueError as e:
            pn.state.notifications.error(str(e), duration=0)

    # ------------------------------------------------------------------------------------------------------------------

    button_name_to_onclick_fn = {
        "import_custom_events": onclickImportCustomEvents,
        "open_label_stores": onclickLabelStores,
        "read_rawData": onclickreaddata,
        "preprocess": onclickpreprocess,
        "psth_computation": onclickpsth,
        "open_visualization": onclickVisualization,
        "open_metadata": onclickMetadata,
        "export_nwb": onclickExportNwb,
    }
    sidebar.attach_callbacks(button_name_to_onclick_fn=button_name_to_onclick_fn)
    sidebar.add_to_template()

    # Expose minimal hooks and widgets to enable programmatic testing
    template._hooks = {
        "onclickVisualization": onclickVisualization,
        "onclickMetadata": onclickMetadata,
        "onclickExportNwb": onclickExportNwb,
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
