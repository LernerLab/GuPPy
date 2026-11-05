"""Panel web application for GuPPy: server startup."""

import panel as pn

from .orchestration.home import build_homepage


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

    Serves the homepage, built anew for each browser session, on one persistent server. Every page
    a step produces is shown inside the homepage. Prints how to open and stop GuPPy, then the
    server's URL once it starts.

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

    def build_page() -> pn.template.BootstrapTemplate:
        return build_homepage(
            input_root_folder=input_root_folder,
            output_root_folder=output_root_folder,
            use_bundled_dandi_cache=use_bundled_dandi_cache,
        )

    print("GuPPy is starting. Open the address below in your browser, and press Ctrl+C here to stop GuPPy.")
    pn.serve(build_page, port=port, address=address, show=open_browser, verbose=True)
