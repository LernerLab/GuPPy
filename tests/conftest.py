import multiprocessing
import os

import holoviews as hv
import panel as pn
import pytest

# Set GUPPY_BASE_DIR before any guppy source modules are imported. This is
# GuPPy's headless flag: orchestration code checks it to skip figure creation.
# The per-test step functions will override this with the actual base_dir.
os.environ.setdefault("GUPPY_BASE_DIR", "1")

# Use "spawn" start method for all multiprocessing in tests. "fork" (the Linux
# default) can deadlock when forking a multi-threaded pytest host, and can stall
# coverage measurement waiting on child-process signals. "spawn" creates a clean
# interpreter for each worker, which is safe in all environments. Windows always
# uses "spawn" so force=True is a no-op there; macOS/Linux benefit from it.
multiprocessing.set_start_method("spawn", force=True)


@pytest.fixture(scope="session")
def panel_extension() -> None:
    """Load the Panel and Holoviews rendering extensions exactly once for the session.

    Panel requires ``pn.extension()`` before any widget instantiation. Holoviews needs a
    plotting backend registered (``hv.extension("bokeh")``) before any ``opts.NdOverlay`` call
    the dashboard/plotter widgets make; registering it here — rather than relying on it being
    loaded implicitly by some earlier test — keeps frontend tests independent of run order.
    """
    pn.extension()
    hv.extension("bokeh")
