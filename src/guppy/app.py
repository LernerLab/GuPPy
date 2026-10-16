"""Panel web application for GuPPy: the route map and server startup."""

import panel as pn

from .orchestration.artifact_view import build_artifact_view
from .orchestration.home import build_homepage
from .orchestration.preprocess_view import build_preprocess_view
from .orchestration.select_artifact_windows import build_select_artifact_windows_view
from .orchestration.step_view import build_page_view
from .orchestration.tonic_analysis import build_tonic_analysis_view
from .orchestration.transients_view import build_transients_view


def serve_app(
    *,
    input_root_folder: str | None = None,
    output_root_folder: str | None = None,
    use_bundled_dandi_cache: bool = True,
    port: int = 0,
    address: str | None = None,
    open_browser: bool = True,
) -> None:
    """Serve the GuPPy application using Panel.

    Serves the homepage plus the step result-view routes and the shared ``page`` route on one
    persistent server (each a per-session factory). Every page GuPPy opens lives on this
    never-torn-down server, so the browser tab is never abruptly disconnected and the whole
    application is reachable through a single port. The server's URL is printed once it starts.

    Parameters
    ----------
    input_root_folder : str or None, optional
        Directory the session folders live under, pre-selected on the homepage.
    output_root_folder : str or None, optional
        Directory the mirrored output tree is written into, pre-selected on the homepage.
    use_bundled_dandi_cache : bool, optional
        Whether the DANDI catalog filter falls back on the dandiset verdicts shipped with GuPPy.
    port : int, optional
        Port the server listens on; 0 picks a free port.
    address : str or None, optional
        Network address the server listens on; ``None`` listens on every interface.
    open_browser : bool, optional
        Whether to open the homepage in a browser on this machine once the server starts.
    """
    routes = {
        "/": lambda: build_homepage(
            input_root_folder=input_root_folder,
            output_root_folder=output_root_folder,
            use_bundled_dandi_cache=use_bundled_dandi_cache,
        ),
        "/page": build_page_view,
        "/preprocess-view": build_preprocess_view,
        "/select-artifact-windows": build_select_artifact_windows_view,
        "/artifact-view": build_artifact_view,
        "/tonic-analysis": build_tonic_analysis_view,
        "/transients-view": build_transients_view,
    }
    pn.serve(routes, port=port, address=address, show=open_browser, verbose=True)
