import os
import socket
import time

import panel as pn
import pytest

from guppy.frontend.frontend_utils import scanPortsAndFind
from guppy.orchestration.home import build_homepage


@pytest.fixture(scope="session")
def ui_base_dir(tmp_path_factory):
    """Provide a real temp directory as GUPPY_BASE_DIR.

    Prevents Tk dialogs from opening when FileSelector widgets initialize.
    """
    base_dir = tmp_path_factory.mktemp("ui_base")
    original = os.environ.get("GUPPY_BASE_DIR")
    os.environ["GUPPY_BASE_DIR"] = str(base_dir)
    yield base_dir
    if original is None:
        del os.environ["GUPPY_BASE_DIR"]
    else:
        os.environ["GUPPY_BASE_DIR"] = original


@pytest.fixture(scope="session")
def live_server(panel_extension, ui_base_dir):
    """Serve the GuPPy homepage via Panel on a free local port.

    Yields the base URL of the running server. Shuts down all Panel servers on
    teardown.
    """
    port = scanPortsAndFind()
    template = build_homepage()
    pn.serve(template, port=port, show=False, threaded=True)

    base_url = f"http://localhost:{port}"

    # Brief poll to confirm the server accepted connections before yielding
    for _ in range(50):
        try:
            connection = socket.create_connection(("localhost", port), timeout=0.1)
            connection.close()
            break
        except OSError:
            time.sleep(0.05)

    yield base_url

    pn.state.kill_all_servers()


@pytest.fixture(scope="session")
def live_server_url(live_server):
    """Return the base URL of the running Panel server."""
    return live_server
