"""Panel web application for GuPPy: server startup."""

import panel as pn

from .orchestration.home import build_homepage

# Bokeh's query argument naming the server session a page attaches to.
SESSION_ID_QUERY_ARGUMENT = "bokeh-session-id"


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

    # Panel builds a page per browser session only from a plain function; any other callable,
    # such as a functools.partial, is served as an object and shown as its repr.
    def build_page() -> pn.template.BootstrapTemplate:
        return build_session_homepage(
            input_root_folder=input_root_folder,
            output_root_folder=output_root_folder,
            use_bundled_dandi_cache=use_bundled_dandi_cache,
        )

    print("GuPPy is starting. Open the address below in your browser, and press Ctrl+C here to stop GuPPy.")
    pn.serve(build_page, port=port, address=address, show=open_browser, verbose=True)


def build_session_homepage(
    *,
    input_root_folder: str | None = None,
    output_root_folder: str | None = None,
    use_bundled_dandi_cache: bool = True,
) -> pn.template.BootstrapTemplate:
    """Build the homepage for one browser session, which a refresh of the page keeps.

    Once the page loads, its address names its server session, so refreshing the browser
    reattaches to that session with its open tabs and form values instead of starting a new one.

    Parameters
    ----------
    input_root_folder : str or None, optional
        Directory the session folders live under, pre-selected on the homepage.
    output_root_folder : str or None, optional
        Directory the mirrored output tree is written into, pre-selected on the homepage.
    use_bundled_dandi_cache : bool, optional
        Whether the DANDI catalog filter falls back on the dandiset verdicts shipped with GuPPy.

    Returns
    -------
    template : pn.template.BootstrapTemplate
        The homepage.
    """
    template = build_homepage(
        input_root_folder=input_root_folder,
        output_root_folder=output_root_folder,
        use_bundled_dandi_cache=use_bundled_dandi_cache,
    )
    pn.state.onload(_name_session_in_address)
    return template


def _name_session_in_address() -> None:
    session_id = pn.state.curdoc.session_context.id
    pn.state.location.update_query(**{SESSION_ID_QUERY_ARGUMENT: session_id})
