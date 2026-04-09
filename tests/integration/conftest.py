import glob
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import holoviews as hv
import pytest

from guppy.frontend.visualization_dashboard import VisualizationDashboard
from guppy.testing.api import step2, step3, step4, step5, step6

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUBBED_TESTING_DATA = PROJECT_ROOT / "stubbed_testing_data"


REPRESENTATIVE_SESSIONS = {
    "csv": {
        "session_subdir": "csv/sample_data_csv_1",
        "storenames_map": {
            "Sample_Control_Channel": "control_region",
            "Sample_Signal_Channel": "signal_region",
            "Sample_TTL": "ttl",
        },
        "npm_timestamp_column_names": None,
        "npm_time_units": None,
        "npm_split_events": [True, True],
    },
    "tdt": {
        "session_subdir": "tdt/Photo_63_207-181030-103332",
        "storenames_map": {
            "Dv1A": "control_dms",
            "Dv2A": "signal_dms",
            "PrtN": "port_entries_dms",
        },
        "npm_timestamp_column_names": None,
        "npm_time_units": None,
        "npm_split_events": [True, True],
    },
    "npm": {
        "session_subdir": "npm/sampleData_NPM_5",
        "storenames_map": {
            "file0_chev1": "control_region1",
            "file0_chod1": "signal_region1",
            "event0": "ttl_region1",
        },
        "npm_timestamp_column_names": None,
        "npm_time_units": None,
        "npm_split_events": None,
    },
    "doric": {
        "session_subdir": "doric/sample_doric_1",
        "storenames_map": {
            "AIn-1 - Raw": "control_region",
            "AIn-2 - Raw": "signal_region",
            "DI--O-1": "ttl",
        },
        "npm_timestamp_column_names": None,
        "npm_time_units": None,
        "npm_split_events": [True, True],
    },
    "nwb": {
        "session_subdir": "nwb/mock_nwbfile",
        "storenames_map": {
            "fiber_photometry_response_series_0": "control_region",
            "fiber_photometry_response_series_1": "signal_region",
            "events": "ttl",
        },
        "npm_timestamp_column_names": None,
        "npm_time_units": None,
        "npm_split_events": [True, True],
    },
}


def _locate_output_directory(*, session_copy: str) -> str:
    session_name = os.path.basename(session_copy)
    output_directories = sorted(glob.glob(os.path.join(session_copy, f"{session_name}_output_*")))
    assert output_directories, f"No output directories found in {session_copy}"

    for output_directory in output_directories:
        if os.path.exists(os.path.join(output_directory, "storesList.csv")):
            return output_directory

    raise AssertionError(f"No storesList.csv found in any output directory under {session_copy}")


def _prepare_pipeline_state(
    *, tmp_path_factory: pytest.TempPathFactory, modality: str
) -> dict[str, str | list[bool] | None]:
    representative_config = REPRESENTATIVE_SESSIONS[modality]
    source_session = os.path.join(str(STUBBED_TESTING_DATA), representative_config["session_subdir"])
    assert os.path.isdir(source_session), f"Sample data not available at expected path: {source_session}"

    temporary_base_directory = tmp_path_factory.mktemp(f"integration_{modality}")
    session_name = os.path.basename(source_session)
    session_copy_path = temporary_base_directory / session_name
    shutil.copytree(source_session, session_copy_path)

    for output_directory in glob.glob(os.path.join(session_copy_path, f"{session_name}_output_*")):
        assert os.path.isdir(
            output_directory
        ), f"Expected output directory for cleanup, got non-directory: {output_directory}"
        shutil.rmtree(output_directory)

    parameters_path = session_copy_path / "GuPPyParamtersUsed.json"
    if parameters_path.exists():
        parameters_path.unlink()

    return {
        "base_directory": str(temporary_base_directory),
        "session_copy": str(session_copy_path),
        "storenames_map": representative_config["storenames_map"],
        "npm_timestamp_column_names": representative_config["npm_timestamp_column_names"],
        "npm_time_units": representative_config["npm_time_units"],
        "npm_split_events": representative_config["npm_split_events"],
    }


def _run_step2(*, pipeline_state: dict[str, str | list[bool] | None]) -> dict[str, str | list[bool] | None]:
    step2(
        base_dir=str(pipeline_state["base_directory"]),
        selected_folders=[str(pipeline_state["session_copy"])],
        storenames_map=pipeline_state["storenames_map"],
        npm_timestamp_column_names=pipeline_state["npm_timestamp_column_names"],
        npm_time_units=pipeline_state["npm_time_units"],
        npm_split_events=pipeline_state["npm_split_events"],
    )
    pipeline_state["output_directory"] = _locate_output_directory(session_copy=str(pipeline_state["session_copy"]))
    return pipeline_state


def _run_step3(*, pipeline_state: dict[str, str | list[bool] | None]) -> dict[str, str | list[bool] | None]:
    step3(
        base_dir=str(pipeline_state["base_directory"]),
        selected_folders=[str(pipeline_state["session_copy"])],
        npm_timestamp_column_names=pipeline_state["npm_timestamp_column_names"],
        npm_time_units=pipeline_state["npm_time_units"],
        npm_split_events=pipeline_state["npm_split_events"],
    )
    pipeline_state["output_directory"] = _locate_output_directory(session_copy=str(pipeline_state["session_copy"]))
    return pipeline_state


def _run_step4(*, pipeline_state: dict[str, str | list[bool] | None]) -> dict[str, str | list[bool] | None]:
    step4(
        base_dir=str(pipeline_state["base_directory"]),
        selected_folders=[str(pipeline_state["session_copy"])],
        npm_timestamp_column_names=pipeline_state["npm_timestamp_column_names"],
        npm_time_units=pipeline_state["npm_time_units"],
        npm_split_events=pipeline_state["npm_split_events"],
    )
    pipeline_state["output_directory"] = _locate_output_directory(session_copy=str(pipeline_state["session_copy"]))
    return pipeline_state


@pytest.fixture(scope="session")
def step2_output_csv(tmp_path_factory: pytest.TempPathFactory):
    pipeline_state = _prepare_pipeline_state(tmp_path_factory=tmp_path_factory, modality="csv")
    return _run_step2(pipeline_state=pipeline_state)


@pytest.fixture(scope="session")
def step2_output_tdt(tmp_path_factory: pytest.TempPathFactory):
    pipeline_state = _prepare_pipeline_state(tmp_path_factory=tmp_path_factory, modality="tdt")
    return _run_step2(pipeline_state=pipeline_state)


@pytest.fixture(scope="session")
def step2_output_npm(tmp_path_factory: pytest.TempPathFactory):
    pipeline_state = _prepare_pipeline_state(tmp_path_factory=tmp_path_factory, modality="npm")
    return _run_step2(pipeline_state=pipeline_state)


@pytest.fixture(scope="session")
def step2_output_doric(tmp_path_factory: pytest.TempPathFactory):
    pipeline_state = _prepare_pipeline_state(tmp_path_factory=tmp_path_factory, modality="doric")
    return _run_step2(pipeline_state=pipeline_state)


@pytest.fixture(scope="session")
def step2_output_nwb(tmp_path_factory: pytest.TempPathFactory):
    pipeline_state = _prepare_pipeline_state(tmp_path_factory=tmp_path_factory, modality="nwb")
    return _run_step2(pipeline_state=pipeline_state)


@pytest.fixture(scope="session")
def step3_output_csv(step2_output_csv):
    return _run_step3(pipeline_state=step2_output_csv)


@pytest.fixture(scope="session")
def step3_output_tdt(step2_output_tdt):
    return _run_step3(pipeline_state=step2_output_tdt)


@pytest.fixture(scope="session")
def step3_output_npm(step2_output_npm):
    return _run_step3(pipeline_state=step2_output_npm)


@pytest.fixture(scope="session")
def step3_output_doric(step2_output_doric):
    return _run_step3(pipeline_state=step2_output_doric)


@pytest.fixture(scope="session")
def step3_output_nwb(step2_output_nwb):
    return _run_step3(pipeline_state=step2_output_nwb)


@pytest.fixture(scope="session")
def step4_output_csv(step3_output_csv):
    return _run_step4(pipeline_state=step3_output_csv)


@pytest.fixture(scope="session")
def step4_output_tdt(step3_output_tdt):
    return _run_step4(pipeline_state=step3_output_tdt)


@pytest.fixture(scope="session")
def step4_output_npm(step3_output_npm):
    return _run_step4(pipeline_state=step3_output_npm)


@pytest.fixture(scope="session")
def step4_output_doric(step3_output_doric):
    return _run_step4(pipeline_state=step3_output_doric)


@pytest.fixture(scope="session")
def step4_output_nwb(step3_output_nwb):
    return _run_step4(pipeline_state=step3_output_nwb)


def _run_step5(*, pipeline_state: dict[str, str | list[bool] | None]) -> dict[str, str | list[bool] | None]:
    step5(
        base_dir=str(pipeline_state["base_directory"]),
        selected_folders=[str(pipeline_state["session_copy"])],
        npm_timestamp_column_names=pipeline_state["npm_timestamp_column_names"],
        npm_time_units=pipeline_state["npm_time_units"],
        npm_split_events=pipeline_state["npm_split_events"],
    )
    pipeline_state["output_directory"] = _locate_output_directory(session_copy=str(pipeline_state["session_copy"]))
    return pipeline_state


def _run_step6(*, pipeline_state: dict[str, str | list[bool] | None]) -> dict[str, str | list[bool] | None]:
    # ParameterizedPlotter uses holoviews opts (e.g. opts.NdOverlay) in reactive methods that Panel
    # evaluates eagerly during VisualizationDashboard.__init__. The Bokeh backend must be registered
    # before instantiation or holoviews raises AttributeError.
    hv.extension("bokeh")
    captured_dashboards: list[VisualizationDashboard] = []
    original_init = VisualizationDashboard.__init__

    def capturing_init(self, *, plotter, basename):
        original_init(self, plotter=plotter, basename=basename)
        captured_dashboards.append(self)

    with patch.object(VisualizationDashboard, "__init__", capturing_init):
        with patch.object(VisualizationDashboard, "show", lambda self: None):
            step6(
                base_dir=str(pipeline_state["base_directory"]),
                selected_folders=[str(pipeline_state["session_copy"])],
                npm_timestamp_column_names=pipeline_state["npm_timestamp_column_names"],
                npm_time_units=pipeline_state["npm_time_units"],
                npm_split_events=pipeline_state["npm_split_events"],
            )

    pipeline_state["captured_dashboards"] = captured_dashboards
    return pipeline_state


@pytest.fixture(scope="session")
def step5_output_csv(step4_output_csv):
    return _run_step5(pipeline_state=step4_output_csv)


@pytest.fixture(scope="session")
def step5_output_tdt(step4_output_tdt):
    return _run_step5(pipeline_state=step4_output_tdt)


@pytest.fixture(scope="session")
def step5_output_npm(step4_output_npm):
    return _run_step5(pipeline_state=step4_output_npm)


@pytest.fixture(scope="session")
def step5_output_doric(step4_output_doric):
    return _run_step5(pipeline_state=step4_output_doric)


@pytest.fixture(scope="session")
def step5_output_nwb(step4_output_nwb):
    return _run_step5(pipeline_state=step4_output_nwb)


@pytest.fixture(scope="session")
def step6_output_csv(step5_output_csv):
    return _run_step6(pipeline_state=step5_output_csv)


@pytest.fixture(scope="session")
def step6_output_tdt(step5_output_tdt):
    return _run_step6(pipeline_state=step5_output_tdt)


@pytest.fixture(scope="session")
def step6_output_npm(step5_output_npm):
    return _run_step6(pipeline_state=step5_output_npm)


@pytest.fixture(scope="session")
def step6_output_doric(step5_output_doric):
    return _run_step6(pipeline_state=step5_output_doric)


@pytest.fixture(scope="session")
def step6_output_nwb(step5_output_nwb):
    return _run_step6(pipeline_state=step5_output_nwb)
