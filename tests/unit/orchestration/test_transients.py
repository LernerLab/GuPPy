import pytest

from guppy.orchestration.transients import (
    execute_average_for_group,
    executeFindFreqAndAmp,
)


@pytest.fixture
def transient_params(base_input_parameters):
    base_input_parameters["numberOfCores"] = 1
    base_input_parameters["moving_window"] = 15
    base_input_parameters["group_session_folders"] = ["/group_1"]
    base_input_parameters["session_folders"] = ["/session_1"]
    return base_input_parameters


@pytest.fixture
def capture_dispatch(monkeypatch):
    """Capture which compute routine executeFindFreqAndAmp dispatches to (none of them serve)."""
    calls = {}
    monkeypatch.setattr(
        "guppy.orchestration.transients.execute_find_freq_and_amp",
        lambda ip, sf, mw, procs: calls.setdefault("individual", sf),
    )
    monkeypatch.setattr(
        "guppy.orchestration.transients.execute_find_freq_and_amp_combined",
        lambda ip, sf, mw, procs: calls.setdefault("combined", sf),
    )
    monkeypatch.setattr(
        "guppy.orchestration.transients.execute_average_for_group",
        lambda ip, folders: calls.setdefault("average", folders),
    )
    return calls


class TestExecuteFindFreqAndAmpDispatch:
    def test_individual_path(self, transient_params, capture_dispatch):
        transient_params["averageForGroup"] = False
        transient_params["combine_data"] = False
        executeFindFreqAndAmp(transient_params)
        assert set(capture_dispatch) == {"individual"}
        assert capture_dispatch["individual"] == ["/session_1"]

    def test_combined_path(self, transient_params, capture_dispatch):
        transient_params["averageForGroup"] = False
        transient_params["combine_data"] = True
        executeFindFreqAndAmp(transient_params)
        assert set(capture_dispatch) == {"combined"}
        assert capture_dispatch["combined"] == ["/session_1"]

    def test_average_path(self, transient_params, capture_dispatch):
        transient_params["averageForGroup"] = True
        transient_params["combine_data"] = False
        executeFindFreqAndAmp(transient_params)
        assert set(capture_dispatch) == {"average"}
        assert capture_dispatch["average"] == ["/group_1"]


def test_execute_average_for_group_raises_for_empty_folders(base_input_parameters):
    with pytest.raises(ValueError, match="No folders selected for group averaging"):
        execute_average_for_group(base_input_parameters, [])
