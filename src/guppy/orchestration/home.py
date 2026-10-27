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
from ..frontend.page_area import PageArea
from ..frontend.sidebar import Sidebar
from ..utils import page_links
from ..utils.progress import StepProgress, _current_step

logger = logging.getLogger(__name__)


def build_homepage(
    *,
    input_root_folder: str | None = None,
    output_root_folder: str | None = None,
    use_bundled_dandi_cache: bool = True,
) -> pn.template.BootstrapTemplate:
    """
    Build and return the GuPPy Panel web-application template.

    Parameters
    ----------
    input_root_folder : str or None, optional
        Directory the session folders live under, pre-selected in the form.
    output_root_folder : str or None, optional
        Directory the mirrored output tree is written into, pre-selected in the form.
    use_bundled_dandi_cache : bool, optional
        Whether the DANDI catalog filter falls back on the dandiset verdicts shipped with GuPPy.

    Returns
    -------
    template : pn.template.BootstrapTemplate
        Fully wired Panel template ready to be served or shown.
    """
    # Naming the extensions here keeps their JavaScript in every session's page, not only the first
    # one served, since a session's pn.extension call replaces the extensions declared before it.
    # Every page a step opens is shown inside this one, so it names what those pages use too.
    pn.extension("tabulator", "codeeditor", notifications=True)
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
    parameter_form = ParameterForm(
        template=template,
        input_root_folder=input_root_folder,
        output_root_folder=output_root_folder,
        use_bundled_dandi_cache=use_bundled_dandi_cache,
    )
    sidebar = Sidebar(template=template)
    page_area = PageArea(home=pn.Column(*template.main.objects, sizing_mode="stretch_width"))
    template.main[:] = [page_area.tabs]

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

    def _collect_page(
        produce_pages: Callable[[], None],
        *,
        title: str | None = None,
        plural_noun: str | None = None,
    ) -> page_links.Page | None:
        # The one page produce_pages offers, or None when it offered none. Pages offered one per
        # session or run are combined into one page, an in-page tab each, named by title and
        # plural_noun.
        with page_links.collect() as pages:
            produce_pages()
        if not pages:
            return None
        if len(pages) == 1:
            (page,) = pages
            name = page.name if title is None else f"{title} — {page.name}"
            return page_links.Page(name=name, content=page.content)
        # Only the visible tab renders, so a dozen dashboards do not all draw at once.
        tabs = pn.Tabs(*[(page.name, page.content) for page in pages], dynamic=True, sizing_mode="stretch_width")
        return page_links.Page(name=f"{title} — {len(pages)} {plural_noun}", content=tabs)

    def _show(key: str, page: page_links.Page | None) -> None:
        if page is not None:
            page_area.show(key=key, name=page.name, content=page.content)

    def _opens_page(button_name: str, open_page: Callable[[], page_links.Page | None]) -> Callable[[object], None]:
        # A click handler that shows the page open_page returns, if any, on the homepage.
        def onclick(event: object = None) -> None:
            _show(button_name, open_page())

        return onclick

    def _run_worker_with_progress(
        worker: Callable[[dict[str, object]], None],
        progress_widget: pn.indicators.Progress,
        *,
        add_curr_dir: bool = False,
        require_selected_outputs: bool = True,
        on_success: Callable[[dict[str, object]], None] | None = None,
        results_button: str | None = None,
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
        if results_button is not None:
            # The previous run's results are being overwritten.
            sidebar.show_results(button_name=results_button, open_results=None)

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
                        # e.g. offer the step's result view once compute succeeded.
                        on_success(inputParameters)

        poller["callback"] = pn.state.add_periodic_callback(poll, period=100)

    def _open_custom_events() -> page_links.Page | None:
        inputParameters = _getInputParametersOrNotify()
        if inputParameters is None:
            return None
        return _collect_page(
            lambda: orchestrate_custom_events_page(inputParameters),
            title="Import Custom Events",
            plural_noun="sessions",
        )

    def _open_label_stores() -> page_links.Page | None:
        inputParameters = _getInputParametersOrNotify()
        if inputParameters is None:
            return None
        # Each Save on a Label Stores page creates a run folder; re-list the runs so it appears here.
        return _collect_page(
            lambda: orchestrate_store_labeling_page(
                inputParameters, on_saved=parameter_form.refresh_individual_outputs
            ),
            title="Label Stores",
            plural_noun="sessions",
        )

    def _open_visualization() -> page_links.Page | None:
        # Unlike steps 2-4, visualization can run on groups alone, so it does not require
        # every selected session to have a run picked. visualizeResults reports what is
        # actually missing.
        inputParameters = _getInputParametersOrNotify()
        if inputParameters is None:
            return None
        try:
            return _collect_page(
                lambda: visualizeResults(inputParameters),
                title="Visualization",
                plural_noun="dashboards",
            )
        except ValueError as e:
            pn.state.notifications.error(str(e), duration=0)
            return None

    def _open_select_artifact_windows() -> page_links.Page | None:
        inputParameters = _getInputParametersOrNotify(require_selected_outputs=True)
        if inputParameters is None:
            return None
        try:
            return _collect_page(lambda: orchestrate_select_artifact_windows(inputParameters))
        except ValueError as e:
            pn.state.notifications.error(str(e), duration=0)
            return None

    def _open_tonic_analysis() -> page_links.Page | None:
        inputParameters = _getInputParametersOrNotify(require_selected_outputs=True)
        if inputParameters is None:
            return None
        try:
            return _collect_page(lambda: orchestrate_tonic_analysis(inputParameters))
        except ValueError as e:
            pn.state.notifications.error(str(e), duration=0)
            return None

    def _open_label_groups() -> page_links.Page | None:
        inputParameters = _getInputParametersOrNotify()
        if inputParameters is None:
            return None
        page = None
        try:
            page = _collect_page(lambda: orchestrate_group_labeling_page(inputParameters))
        except ValueError as e:
            pn.state.notifications.error(str(e), duration=0)
        # A group defined just now becomes selectable without re-navigating the browser.
        parameter_form.refresh_group_folders()
        return page

    def _open_metadata() -> page_links.Page | None:
        inputParameters = _getInputParametersOrNotify(require_selected_outputs=True)
        if inputParameters is None:
            return None
        try:
            return _collect_page(
                lambda: orchestrate_metadata_page(inputParameters),
                title="Input Metadata",
                plural_noun="sessions",
            )
        except ValueError as e:
            pn.state.notifications.error(str(e), duration=0)
            return None

    onclickImportCustomEvents = _opens_page("import_custom_events", _open_custom_events)
    onclickLabelStores = _opens_page("open_label_stores", _open_label_stores)
    onclickVisualization = _opens_page("open_visualization", _open_visualization)
    onclickSelectArtifactWindows = _opens_page("select_artifact_windows", _open_select_artifact_windows)
    onclickTonicAnalysis = _opens_page("tonic_analysis", _open_tonic_analysis)
    onclickLabelGroups = _opens_page("label_groups", _open_label_groups)
    onclickMetadata = _opens_page("open_metadata", _open_metadata)

    def onclickreaddata(event: object = None) -> None:
        _run_worker_with_progress(run_read_raw_data_step, sidebar.read_progress)

    def onclickpreprocess(event: object = None) -> None:
        def _show_results(inputParameters: dict[str, object]) -> None:
            sidebar.show_results(
                button_name="preprocess",
                open_results=lambda: _show(
                    "preprocess",
                    _collect_page(lambda: open_preprocess_view(inputParameters["session_folders"], inputParameters)),
                ),
            )

        _run_worker_with_progress(
            run_preprocess_step, sidebar.extract_progress, on_success=_show_results, results_button="preprocess"
        )

    def onclickRemoveArtifacts(event: object = None) -> None:
        def _show_results(inputParameters: dict[str, object]) -> None:
            sidebar.show_results(
                button_name="remove_artifacts",
                open_results=lambda: _show(
                    "remove_artifacts",
                    _collect_page(lambda: open_artifact_view(inputParameters["session_folders"], inputParameters)),
                ),
            )

        _run_worker_with_progress(
            run_remove_artifacts_step,
            sidebar.remove_artifacts_progress,
            on_success=_show_results,
            results_button="remove_artifacts",
        )

    def onclickpsth(event: object = None) -> None:
        def _show_results(inputParameters: dict[str, object]) -> None:
            sidebar.show_results(
                button_name="psth_computation",
                open_results=lambda: _show(
                    "psth_computation",
                    _collect_page(lambda: open_transients_view(inputParameters["session_folders"], inputParameters)),
                ),
            )

        _run_worker_with_progress(
            run_psth_step,
            sidebar.psth_progress,
            add_curr_dir=True,
            on_success=_show_results,
            results_button="psth_computation",
        )

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
        "onclickLabelStores": onclickLabelStores,
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
        "dandi_file_panel": parameter_form.dandi_file_panel,
        "read_progress": sidebar.read_progress,
        "extract_progress": sidebar.extract_progress,
        "psth_progress": sidebar.psth_progress,
        "group_progress": sidebar.group_progress,
        "group_folders_selector": parameter_form.group_folders_selector,
        "same_root_checkbox": parameter_form.same_root_checkbox,
        "output_root_selector": parameter_form.output_root_selector,
        "input_root_selector": parameter_form.input_root_selector,
        "remove_artifacts_progress": sidebar.remove_artifacts_progress,
        "results_buttons": sidebar.results_buttons,
        "page_area": page_area,
    }

    return template
