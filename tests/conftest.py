import os
import sys

import matplotlib

# Set GUPPY_BASE_DIR before any guppy source modules are imported. This is
# GuPPy's headless flag: orchestration code checks it to skip figure creation.
# The per-test step functions will override this with the actual base_dir.
os.environ.setdefault("GUPPY_BASE_DIR", "1")

# Ensure the 'src' directory is on sys.path for tests without installation
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Belt-and-suspenders: use the non-interactive Agg backend in case any code
# path still reaches matplotlib figure creation.
matplotlib.use("Agg")
