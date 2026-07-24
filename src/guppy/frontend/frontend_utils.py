import logging
import os
import socket
import threading
from collections.abc import Callable
from random import randint

import panel as pn

logger = logging.getLogger(__name__)


def default_root_path() -> str:
    """Starting directory for the GUI's directory pickers.

    Honors ``GUPPY_BASE_DIR`` (used by headless tests and the testing API to
    point pickers at a temp directory); otherwise falls back to the user's
    home directory.
    """
    base_dir_env = os.environ.get("GUPPY_BASE_DIR")
    if base_dir_env and os.path.isdir(base_dir_env):
        return base_dir_env
    return os.path.expanduser("~")


# Ports that all major browsers refuse to connect to (ERR_UNSAFE_PORT).
# Defined in the WHATWG Fetch Standard (https://fetch.spec.whatwg.org/#bad-port),
# a cross-browser spec maintained by Apple, Google, Mozilla, and Microsoft.
_CHROME_UNSAFE_PORTS = {5060, 5061}


def scanPortsAndFind(start_port: int = 5000, end_port: int = 5200, host: str = "127.0.0.1") -> int:
    """Find an available TCP port by randomly sampling the given range.

    Skips ports that browsers refuse to connect to (Chrome unsafe-port list)
    and ports that are already bound.

    Parameters
    ----------
    start_port : int, optional
        Lower bound of the port range to search (inclusive). Default is 5000.
    end_port : int, optional
        Upper bound of the port range to search (inclusive). Default is 5200.
    host : str, optional
        Host address to probe. Default is ``"127.0.0.1"``.

    Returns
    -------
    int
        An available port number within ``[start_port, end_port]``.
    """
    while True:
        port = randint(start_port, end_port)
        if port in _CHROME_UNSAFE_PORTS:
            continue
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.001)  # Set timeout to avoid long waiting on closed ports
        result = sock.connect_ex((host, port))
        if result == 0:  # If the connection is successful, the port is open
            continue
        else:
            break

    return port


def serve_blocking_page(build_template: Callable[[Callable[[], None]], "pn.template.BaseTemplate"]) -> None:
    """Serve a single Panel page and block until the user finishes with it.

    The preprocessing and transient-analysis steps run as subprocesses with no
    running event loop, so a bare ``template.show()`` would block forever (a Bokeh
    server does not stop when the tab closes). Instead the server runs on a
    background thread and this call waits on a :class:`threading.Event` that is set
    either when the page signals completion (its "continue"/"save" button) or when
    the browser session closes, then stops the server so the pipeline advances.

    Parameters
    ----------
    build_template : callable
        Given an ``on_done`` callback, returns the Panel template to serve. The
        template is expected to invoke ``on_done`` when the user is finished.
    """
    # When PR #397 (tonic epochs) merges, its ``_serve_tonic_epoch_page`` should be
    # refactored to call this shared helper.
    done = threading.Event()

    def build_page() -> "pn.template.BaseTemplate":
        pn.state.on_session_destroyed(lambda session_context: done.set())
        return build_template(done.set)

    port = scanPortsAndFind(start_port=5000, end_port=5200)
    server = pn.serve(build_page, port=port, threaded=True, show=True)
    try:
        done.wait()
    finally:
        server.stop()
