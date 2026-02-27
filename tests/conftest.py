import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import pytest

# Ensure the 'src' directory is on sys.path for tests without installation
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Use the non-interactive Agg backend for the entire test session so no GUI
# windows are ever allocated.
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_matplotlib_figures(monkeypatch):
    # Patch plt.show() to close all figures immediately instead of displaying
    # them. Without this, matplotlib's figure manager retains references to
    # every created figure for the lifetime of the test session, causing OOM
    # crashes on CI.
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: plt.close("all"))
    yield
    # Belt-and-suspenders: close any figures that were created but never shown.
    plt.close("all")
