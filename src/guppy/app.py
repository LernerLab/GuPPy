"""Panel web application for GuPPy: the route map and server startup."""

import panel as pn

from .orchestration.artifact_view import build_artifact_view
from .orchestration.home import build_homepage
from .orchestration.preprocess_view import build_preprocess_view
from .orchestration.select_artifact_windows import build_select_artifact_windows_view
from .orchestration.tonic_analysis import build_tonic_analysis_view
from .orchestration.transients_view import build_transients_view


def serve_app(
    *, start_path: str | None = None, data_root: str | None = None, output_base_directory: str | None = None
) -> None:
    """Serve the GuPPy application using Panel.

    Serves the homepage plus the step result-view routes on one persistent server (each a
    per-session factory). The views live on this never-torn-down server so the browser tab
    is never abruptly disconnected. ``show=True`` opens the homepage at ``/``.

    Parameters
    ----------
    start_path : str or None, optional
        Initial directory shown in the folder-selection widget. When None the widget
        starts in the current working directory.
    data_root : str or None, optional
        Directory the session folders live under, pre-selected on the homepage.
    output_base_directory : str or None, optional
        Directory the mirrored output tree is written into, pre-selected on the homepage.
    """
    routes = {
        "/": lambda: build_homepage(
            start_path=start_path, data_root=data_root, output_base_directory=output_base_directory
        ),
        "/preprocess-view": build_preprocess_view,
        "/select-artifact-windows": build_select_artifact_windows_view,
        "/artifact-view": build_artifact_view,
        "/tonic-analysis": build_tonic_analysis_view,
        "/transients-view": build_transients_view,
    }
    pn.serve(routes, show=True)
