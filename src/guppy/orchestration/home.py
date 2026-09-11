import logging
from collections.abc import Callable
from contextvars import copy_context
from importlib.metadata import version
from pathlib import Path
from threading import Thread

import panel as pn

from .artifact_view import open_artifact_view
from .export_nwb import run_export_nwb_step
from .group_analysis import run_group_analysis_step
from .group_labeling import orchestrate_group_labeling_page
from .import_custom_events import orchestrate_custom_events_page
from .metadata import orchestrate_metadata_page
from .preprocess import run_preprocess_step, run_remove_artifacts_step
from .preprocess_view import open_preprocess_view
from .psth import run_psth_step
from .read_raw_data import run_read_raw_data_step
from .select_artifact_windows import orchestrate_select_artifact_windows
from .store_labeling import orchestrate_store_labeling_page
from .tonic_analysis import orchestrate_tonic_analysis
from .transients_view import open_transients_view
from .visualize import visualizeResults
from ..frontend.input_parameters import ParameterForm
from ..frontend.sidebar import Sidebar
from ..utils.progress import StepProgress, _current_step

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
    current_dir = str(Path.cwd())
    # Guards against launching a second pipeline step while one is still running — the
    # handlers no longer block the IOLoop, so without this a user could start overlapping
    # runs that clobber the shared progress file and output directories.
    step_running = {"active": False}

    template = pn.template.BootstrapTemplate(title="Input Parameters GUI")
    # Right-aligned in the header bar so the running version is visible without scrolling
    # the long sidebar, and is not mistaken for another pipeline step.
    template.header.append(
        pn.pane.Markdown(
            f"GuPPy {version('guppy-neuro')}",
            styles={"color": "white", "margin-left": "auto", "margin-right": "1em"},
        )
    )
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
        require_selected_outputs: bool = True,
        on_success: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        # Shared launch pattern for the pipeline steps that run a worker in a background
        # thread while a progress bar reports what the worker has emitted.
        if step_running["active"]:
            pn.state.notifications.error("A pipeline step is already running; wait for it to finish.", duration=0)
            return
        inputParameters = _getInputParametersOrNotify(require_selected_outputs=require_selected_outputs)
        if inputParameters is None:
            return
        if add_curr_dir:
            inputParameters["curr_dir"] = current_dir

        progress_widget.value = 0
        progress_widget.bar_color = "success"

        # Bind the step's progress sink for the worker only. Running the setter inside the
        # copied context confines the binding to the thread that context carries, leaving
        # this handler's own context untouched.
        step = StepProgress()
        context = copy_context()
        context.run(_current_step.set, step)

        step_running["active"] = True
        thread = Thread(target=context.run, args=(worker, inputParameters))
        thread.start()

        # Poll progress from a periodic callback on the server IOLoop rather than
        # busy-looping and joining the worker here. A synchronous wait would block the
        # IOLoop for the whole step, so the main app would stop answering websocket
        # heartbeats and the browser tab would drop its connection and never recover.
        poller: dict[str, object] = {}

        def poll() -> None:
            progress_widget.max = max(step.total, 1)
            progress_widget.value = min(step.value, progress_widget.max)
            if step.error_message:
                progress_widget.bar_color = "danger"
            # Completion is the worker thread finishing, never the counter reaching its
            # total: a step whose declared total undercounts its real work would otherwise
            # report success partway through and open its result view against half-written
            # output.
            if not thread.is_alive():
                poller["callback"].stop()
                step_running["active"] = False
                for warning in step.warnings:
                    pn.state.notifications.warning(warning, duration=0)
                if step.error_message:
                    pn.state.notifications.error(step.error_message, duration=0)
                else:
                    progress_widget.value = progress_widget.max
                    if on_success is not None:
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

    def onclickVisualization(event: object = None) -> None:
        # Unlike steps 2-4, visualization can run on groups alone, so it does not require
        # every selected session to have a run picked. visualizeResults reports what is
        # actually missing.
        inputParameters = _getInputParametersOrNotify()
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

    def onclickSelectArtifactWindows(event: object = None) -> None:
        inputParameters = _getInputParametersOrNotify(require_selected_outputs=True)
        if inputParameters is None:
            return
        try:
            orchestrate_select_artifact_windows(inputParameters)
        except ValueError as e:
            pn.state.notifications.error(str(e), duration=0)

    def onclickTonicAnalysis(event: object = None) -> None:
        inputParameters = _getInputParametersOrNotify(require_selected_outputs=True)
        if inputParameters is None:
            return
        try:
            orchestrate_tonic_analysis(inputParameters)
        except ValueError as e:
            pn.state.notifications.error(str(e), duration=0)

    def onclickRemoveArtifacts(event: object = None) -> None:
        def _open_view(inputParameters: dict[str, object]) -> None:
            open_artifact_view(inputParameters["session_folders"], inputParameters)

        _run_worker_with_progress(run_remove_artifacts_step, sidebar.remove_artifacts_progress, on_success=_open_view)

    def onclickpsth(event: object = None) -> None:
        def _open_view(inputParameters: dict[str, object]) -> None:
            open_transients_view(inputParameters["session_folders"], inputParameters)

        _run_worker_with_progress(run_psth_step, sidebar.psth_progress, add_curr_dir=True, on_success=_open_view)

    def onclickLabelGroups(event: object = None) -> None:
        inputParameters = _getInputParametersOrNotify()
        if inputParameters is None:
            return
        try:
            orchestrate_group_labeling_page(inputParameters)
        except ValueError as e:
            pn.state.notifications.error(str(e), duration=0)
        # A group defined just now becomes selectable without re-navigating the browser.
        parameter_form.refresh_group_folders()

    def onclickGroupAnalysis(event: object = None) -> None:
        def _list_new_group(inputParameters: dict[str, object]) -> None:
            # Make the group just written selectable for visualization without
            # requiring the user to re-pick the destination directory.
            parameter_form.refresh_group_folders()
            pn.state.notifications.success("Group Analysis complete.")

        # The group step reads the group selectors, not files_1's selected runs.
        _run_worker_with_progress(
            run_group_analysis_step,
            sidebar.group_progress,
            require_selected_outputs=False,
            on_success=_list_new_group,
        )

    def onclickMetadata(event: object = None) -> None:
        inputParameters = _getInputParametersOrNotify(require_selected_outputs=True)
        if inputParameters is None:
            return
        try:
            orchestrate_metadata_page(inputParameters)
        except ValueError as e:
            pn.state.notifications.error(str(e), duration=0)

    def onclickExportNwb(event: object = None) -> None:
        def _notify_exported(inputParameters: dict[str, object]) -> None:
            # The export writes its .nwb files into the output directories and opens no result
            # view, so this notification is the only sign the step finished. A partial failure
            # takes the error branch instead, so this only fires when every session succeeded.
            pn.state.notifications.success("Export to NWB complete.")

        _run_worker_with_progress(run_export_nwb_step, sidebar.export_progress, on_success=_notify_exported)

    # ------------------------------------------------------------------------------------------------------------------

    button_name_to_onclick_fn = {
        "import_custom_events": onclickImportCustomEvents,
        "open_label_stores": onclickLabelStores,
        "read_rawData": onclickreaddata,
        "preprocess": onclickpreprocess,
        "select_artifact_windows": onclickSelectArtifactWindows,
        "remove_artifacts": onclickRemoveArtifacts,
        "tonic_analysis": onclickTonicAnalysis,
        "psth_computation": onclickpsth,
        "label_groups": onclickLabelGroups,
        "group_analysis": onclickGroupAnalysis,
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
        "onclickSelectArtifactWindows": onclickSelectArtifactWindows,
        "onclickRemoveArtifacts": onclickRemoveArtifacts,
        "onclickTonicAnalysis": onclickTonicAnalysis,
        "onclickpsth": onclickpsth,
        "onclickLabelGroups": onclickLabelGroups,
        "onclickGroupAnalysis": onclickGroupAnalysis,
        "getInputParameters": parameter_form.getInputParameters,
    }
    template._widgets = {
        "files_1": parameter_form.files_1,
        "source_mode": parameter_form.source_mode,
        "dandi_selector": parameter_form.dandi_selector,
        "read_progress": sidebar.read_progress,
        "extract_progress": sidebar.extract_progress,
        "psth_progress": sidebar.psth_progress,
        "group_progress": sidebar.group_progress,
        "group_folders_selector": parameter_form.group_folders_selector,
        "remove_artifacts_progress": sidebar.remove_artifacts_progress,
    }

    return template
