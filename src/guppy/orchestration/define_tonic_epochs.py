"""Define Tonic Epochs step: an interactive page served by the persistent main app.

Clicking its sidebar button opens a browser tab on the
main app's ``/define-tonic-epochs`` route (served by :data:`build_define_tonic_epochs_view`),
composed from the Step-3 outputs already on disk. Averaging those traces over the windows is
a read-only reduction, so saving on that page both writes the definitions
(``tonic_epochs_<recording_site>.csv``) and computes the per-epoch means
(``tonic_<recording_site>.h5``) — no further pipeline step is needed. See
``orchestration/step_view.py`` for the shared token/registry/serving plumbing.
"""

import logging

from .step_view import StepView, resolve_run_folders
from ..frontend.tonic_epochs import build_tonic_epoch_page
from ..utils.validation import validate_preprocessing_outputs_present

logger = logging.getLogger(__name__)

_ACTION = "defining tonic epochs"


def _build_page(session_folders: list, inputParameters: dict) -> "object":
    run_folders = resolve_run_folders(session_folders, inputParameters)
    validate_preprocessing_outputs_present(run_folders=run_folders, action=_ACTION)
    return build_tonic_epoch_page(run_folders=run_folders)


_view = StepView(route="define-tonic-epochs", title="GuPPy — Define Tonic Epochs", build_page=_build_page)

# Route factory for main.py's route map, and open hook for home.py.
build_define_tonic_epochs_view = _view.route_factory
open_define_tonic_epochs_view = _view.open


def orchestrate_define_tonic_epochs(inputParameters: dict[str, object]) -> None:
    """Open the Define Tonic Epochs page for the selected sessions.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; uses ``session_folders``, ``selected_runs``,
        and ``combine_data``.

    Raises
    ------
    ValueError
        If any selected run has no Step-3 preprocessing outputs.
    """
    session_folders = inputParameters["session_folders"]
    validate_preprocessing_outputs_present(
        run_folders=resolve_run_folders(session_folders, inputParameters), action=_ACTION
    )
    open_define_tonic_epochs_view(session_folders, inputParameters)
