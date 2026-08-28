import os
import shutil
import types

import pytest

import guppy.testing.api as testing_api
from guppy.orchestration.store_labeling import (
    build_store_labeling_template,
    read_header,
)
from guppy_test_data import STUBBED_TESTING_DATA


class FakeTemplate:
    def __init__(self, *, hooks=None, widgets=None):
        self._hooks = {} if hooks is None else hooks
        self._widgets = {} if widgets is None else widgets


@pytest.fixture
def api_workspace(tmp_path):
    base_directory = tmp_path / "data_root"
    base_directory.mkdir()

    session_directory = base_directory / "session_one"
    session_directory.mkdir()

    other_directory = tmp_path / "other_root"
    other_directory.mkdir()

    foreign_session_directory = other_directory / "session_two"
    foreign_session_directory.mkdir()

    return {
        "base_directory": str(base_directory),
        "session_directory": str(session_directory),
        "foreign_session_directory": str(foreign_session_directory),
        "missing_session_directory": str(base_directory / "missing_session"),
        "missing_base_directory": str(tmp_path / "missing_root"),
    }


@pytest.fixture
def valid_store_id_to_store_label():
    return {"RawSignal": "signal_region"}


class TestSharedApiValidation:
    @pytest.mark.parametrize("step_name", ["step1", "step2", "step3", "step4", "step5"])
    def test_steps_require_existing_base_directory(self, api_workspace, valid_store_id_to_store_label, step_name):
        step = getattr(testing_api, step_name)
        kwargs = {
            "base_dir": api_workspace["missing_base_directory"],
            "selected_folders": [api_workspace["session_directory"]],
        }
        if step_name == "step1":
            kwargs["store_id_to_store_label"] = valid_store_id_to_store_label
        if step_name in ("step2", "step3", "step4", "step5"):
            kwargs["selected_runs"] = {api_workspace["session_directory"]: ["1"]}

        with pytest.raises(ValueError, match="base_dir does not exist or is not a directory"):
            step(**kwargs)

    @pytest.mark.parametrize("step_name", ["step1", "step2", "step3", "step4", "step5"])
    def test_steps_require_selected_folders(self, api_workspace, valid_store_id_to_store_label, step_name):
        step = getattr(testing_api, step_name)
        kwargs = {
            "base_dir": api_workspace["base_directory"],
            "selected_folders": [],
        }
        if step_name == "step1":
            kwargs["store_id_to_store_label"] = valid_store_id_to_store_label
        if step_name in ("step2", "step3", "step4", "step5"):
            kwargs["selected_runs"] = {}

        with pytest.raises(ValueError, match="selected_folders must be a non-empty iterable"):
            step(**kwargs)

    @pytest.mark.parametrize("step_name", ["step1", "step2", "step3", "step4", "step5"])
    def test_steps_reject_missing_session_directory(self, api_workspace, valid_store_id_to_store_label, step_name):
        step = getattr(testing_api, step_name)
        kwargs = {
            "base_dir": api_workspace["base_directory"],
            "selected_folders": [api_workspace["missing_session_directory"]],
        }
        if step_name == "step1":
            kwargs["store_id_to_store_label"] = valid_store_id_to_store_label
        if step_name in ("step2", "step3", "step4", "step5"):
            kwargs["selected_runs"] = {api_workspace["missing_session_directory"]: ["1"]}

        with pytest.raises(ValueError, match="Session path does not exist or is not a directory"):
            step(**kwargs)

    @pytest.mark.parametrize("step_name", ["step1", "step2", "step3", "step4", "step5"])
    def test_steps_require_sessions_under_base_directory(self, api_workspace, valid_store_id_to_store_label, step_name):
        step = getattr(testing_api, step_name)
        kwargs = {
            "base_dir": api_workspace["base_directory"],
            "selected_folders": [api_workspace["foreign_session_directory"]],
        }
        if step_name == "step1":
            kwargs["store_id_to_store_label"] = valid_store_id_to_store_label
        if step_name in ("step2", "step3", "step4", "step5"):
            kwargs["selected_runs"] = {api_workspace["foreign_session_directory"]: ["1"]}

        with pytest.raises(ValueError, match="must share the same parent equal to base_dir"):
            step(**kwargs)


class TestStep1Validation:
    @pytest.mark.parametrize(
        ("store_id_to_store_label", "message"),
        [
            ({}, "store_id_to_store_label must be a non-empty dict"),
            ({"": "signal_region"}, "Invalid store_id key"),
            ({"RawSignal": ""}, "Invalid store_label"),
            ({1: "signal_region"}, "Invalid store_id key"),
            ({"RawSignal": None}, "Invalid store_label"),
        ],
    )
    def test_step1_validates_store_id_to_store_label(self, api_workspace, store_id_to_store_label, message):
        with pytest.raises(ValueError, match=message):
            testing_api.step1(
                base_dir=api_workspace["base_directory"],
                selected_folders=[api_workspace["session_directory"]],
                store_id_to_store_label=store_id_to_store_label,
            )

    def test_step1_validates_run_name_policy(self, api_workspace, valid_store_id_to_store_label):
        with pytest.raises(ValueError, match="run_name_policy must be 'create' or 'overwrite'"):
            testing_api.step1(
                base_dir=api_workspace["base_directory"],
                selected_folders=[api_workspace["session_directory"]],
                store_id_to_store_label=valid_store_id_to_store_label,
                run_name_policy="bogus",
            )


class TestApiRuntimeErrors:
    @pytest.mark.parametrize("step_name", ["step1", "step2", "step3", "step4", "step5"])
    def test_steps_require_get_input_parameters_hook(
        self, api_workspace, valid_store_id_to_store_label, monkeypatch, step_name
    ):
        monkeypatch.setattr(
            testing_api,
            "build_homepage",
            lambda **kwargs: FakeTemplate(widgets={"files_1": types.SimpleNamespace(value=None)}),
        )

        step = getattr(testing_api, step_name)
        kwargs = {
            "base_dir": api_workspace["base_directory"],
            "selected_folders": [api_workspace["session_directory"]],
        }
        if step_name == "step1":
            kwargs["store_id_to_store_label"] = valid_store_id_to_store_label
        if step_name in ("step2", "step3", "step4", "step5"):
            kwargs["selected_runs"] = {api_workspace["session_directory"]: ["1"]}

        with pytest.raises(RuntimeError, match="getInputParameters"):
            step(**kwargs)

    @pytest.mark.parametrize("step_name", ["step1", "step2", "step3", "step4", "step5"])
    def test_steps_require_files_widget(self, api_workspace, valid_store_id_to_store_label, monkeypatch, step_name):
        monkeypatch.setattr(
            testing_api,
            "build_homepage",
            lambda **kwargs: FakeTemplate(hooks={"getInputParameters": lambda: {}}),
        )

        step = getattr(testing_api, step_name)
        kwargs = {
            "base_dir": api_workspace["base_directory"],
            "selected_folders": [api_workspace["session_directory"]],
        }
        if step_name == "step1":
            kwargs["store_id_to_store_label"] = valid_store_id_to_store_label
        if step_name in ("step2", "step3", "step4", "step5"):
            kwargs["selected_runs"] = {api_workspace["session_directory"]: ["1"]}

        with pytest.raises(RuntimeError, match="files_1"):
            step(**kwargs)

    def test_save_parameters_requires_get_input_parameters_hook(self, api_workspace, monkeypatch):
        monkeypatch.setattr(
            testing_api,
            "build_homepage",
            lambda **kwargs: FakeTemplate(widgets={"files_1": types.SimpleNamespace(value=None)}),
        )

        with pytest.raises(RuntimeError, match="getInputParameters"):
            testing_api.save_parameters_snapshot(
                base_dir=api_workspace["base_directory"],
                selected_folders=[api_workspace["session_directory"]],
            )

    def test_save_parameters_requires_files_widget(self, api_workspace, monkeypatch):
        monkeypatch.setattr(
            testing_api,
            "build_homepage",
            lambda **kwargs: FakeTemplate(hooks={"getInputParameters": lambda: {}}),
        )

        with pytest.raises(RuntimeError, match="files_1"):
            testing_api.save_parameters_snapshot(
                base_dir=api_workspace["base_directory"],
                selected_folders=[api_workspace["session_directory"]],
            )


class TestNormalizeSelectedRuns:
    def test_raises_when_not_a_dict(self):
        with pytest.raises(ValueError, match="must be a dict"):
            testing_api._normalize_selected_runs(["not", "a", "dict"], ["/abs/session"])

    def test_raises_when_session_key_not_in_selected_folders(self, tmp_path):
        session = tmp_path / "session"
        session.mkdir()
        foreign = tmp_path / "foreign"
        foreign.mkdir()
        with pytest.raises(ValueError, match="not in selected_folders"):
            testing_api._normalize_selected_runs(
                {str(foreign): ["run1"]},
                [str(session)],
            )

    def test_raises_when_run_names_not_a_list(self, tmp_path):
        session = tmp_path / "session"
        session.mkdir()
        with pytest.raises(ValueError, match="non-empty list of non-empty strings"):
            testing_api._normalize_selected_runs(
                {str(session): "run1"},
                [str(session)],
            )

    def test_raises_when_run_names_list_is_empty(self, tmp_path):
        session = tmp_path / "session"
        session.mkdir()
        with pytest.raises(ValueError, match="non-empty list of non-empty strings"):
            testing_api._normalize_selected_runs(
                {str(session): []},
                [str(session)],
            )

    def test_raises_when_run_name_is_empty_string(self, tmp_path):
        session = tmp_path / "session"
        session.mkdir()
        with pytest.raises(ValueError, match="non-empty list of non-empty strings"):
            testing_api._normalize_selected_runs(
                {str(session): [""]},
                [str(session)],
            )

    def test_raises_when_run_name_is_not_a_string(self, tmp_path):
        session = tmp_path / "session"
        session.mkdir()
        with pytest.raises(ValueError, match="non-empty list of non-empty strings"):
            testing_api._normalize_selected_runs(
                {str(session): [123]},
                [str(session)],
            )

    def test_raises_when_session_missing_from_mapping(self, tmp_path):
        session_a = tmp_path / "sessionA"
        session_a.mkdir()
        session_b = tmp_path / "sessionB"
        session_b.mkdir()
        with pytest.raises(ValueError, match="missing entries for sessions"):
            testing_api._normalize_selected_runs(
                {str(session_a): ["run1"]},
                [str(session_a), str(session_b)],
            )

    def test_uses_custom_parameter_name_in_error(self, tmp_path):
        with pytest.raises(ValueError, match="custom_param must be a dict"):
            testing_api._normalize_selected_runs(
                "bad",
                [str(tmp_path)],
                parameter_name="custom_param",
            )


class TestParseStoreLabel:
    @pytest.mark.parametrize(
        "store_label, expected",
        [
            ("signal_DMS", ("signal", "DMS")),
            ("control_DMS", ("control", "DMS")),
            ("covariate_akinesia", ("behavioral covariate", "akinesia")),
            ("ttl", ("event TTLs", "ttl")),
            ("port_entries_dms", ("event TTLs", "port_entries_dms")),
        ],
    )
    def test_grammar(self, store_label, expected):
        assert testing_api._parse_store_label(store_label=store_label) == expected


@pytest.fixture
def staged_csv_session(tmp_path):
    """Copy the csv stub session into a temporary workspace, without any prior outputs."""
    base_directory = tmp_path / "data_root"
    base_directory.mkdir()
    session_copy = base_directory / "sample_data_csv_1"
    shutil.copytree(
        os.path.join(str(STUBBED_TESTING_DATA), "csv", "sample_data_csv_1"),
        session_copy,
        ignore=shutil.ignore_patterns("sample_data_csv_1_output_*", "GuPPyParamtersUsed.json"),
    )
    return {"base_dir": str(base_directory), "session": str(session_copy)}


class TestStep1Driver:
    def test_unknown_store_id_raises(self, staged_csv_session):
        with pytest.raises(ValueError, match="not discovered"):
            testing_api.step1(
                base_dir=staged_csv_session["base_dir"],
                selected_folders=[staged_csv_session["session"]],
                store_id_to_store_label={"Nonexistent_Channel": "signal_region"},
            )

    def test_control_without_matching_signal_raises(self, staged_csv_session):
        with pytest.raises(ValueError, match="no matching 'signal_region'"):
            testing_api.step1(
                base_dir=staged_csv_session["base_dir"],
                selected_folders=[staged_csv_session["session"]],
                store_id_to_store_label={"Sample_Control_Channel": "control_region"},
            )

    def test_npm_kwargs_on_non_npm_session_raise(self, staged_csv_session):
        with pytest.raises(ValueError, match="contains no NPM data"):
            testing_api.step1(
                base_dir=staged_csv_session["base_dir"],
                selected_folders=[staged_csv_session["session"]],
                store_id_to_store_label={
                    "Sample_Control_Channel": "control_region",
                    "Sample_Signal_Channel": "signal_region",
                },
                npm_split_events=[False, True],
            )

    def test_signal_without_control_rejected_under_isosbestic_control(self, staged_csv_session):
        with pytest.raises(ValueError, match="signals have no control"):
            testing_api.step1(
                base_dir=staged_csv_session["base_dir"],
                selected_folders=[staged_csv_session["session"]],
                store_id_to_store_label={"Sample_Signal_Channel": "signal_region", "Sample_TTL": "ttl"},
            )

    def test_signal_without_control_accepted_without_isosbestic_control(self, staged_csv_session):
        testing_api.step1(
            base_dir=staged_csv_session["base_dir"],
            selected_folders=[staged_csv_session["session"]],
            store_id_to_store_label={"Sample_Signal_Channel": "signal_region", "Sample_TTL": "ttl"},
            isosbestic_control=False,
        )

        stores_list_path = os.path.join(staged_csv_session["session"], "sample_data_csv_1_output_1", "storesList.csv")
        assert os.path.exists(stores_list_path)


@pytest.fixture
def npm_template_two_timestamp_columns(panel_extension):
    """Label Stores template for the NPM_3 stub: two timestamp columns, split checkbox on file 1."""
    folder_path = os.path.join(str(STUBBED_TESTING_DATA), "npm", "sampleData_NPM_3")
    input_parameters = {"noChannels": 2}
    _, _, npm_interactive = read_header(input_parameters, 2, folder_path)
    return build_store_labeling_template(
        [], [], folder_path, inputParameters=input_parameters, npm_interactive=npm_interactive
    )


@pytest.fixture
def npm_template_single_timestamp_column(panel_extension):
    """Label Stores template for the NPM_4 stub: one timestamp column, split checkbox on file 1."""
    folder_path = os.path.join(str(STUBBED_TESTING_DATA), "npm", "sampleData_NPM_4")
    input_parameters = {"noChannels": 2}
    _, _, npm_interactive = read_header(input_parameters, 2, folder_path)
    return build_store_labeling_template(
        [], [], folder_path, inputParameters=input_parameters, npm_interactive=npm_interactive
    )


class TestDriveNpmConfigurationForm:
    def test_split_events_length_mismatch_raises(self, npm_template_two_timestamp_columns):
        with pytest.raises(ValueError, match="one boolean per file"):
            testing_api._drive_npm_configuration_form(
                template=npm_template_two_timestamp_columns,
                npm_timestamp_column_name=None,
                npm_time_unit=None,
                npm_split_events=[True],
            )

    def test_split_true_without_checkbox_raises(self, npm_template_two_timestamp_columns):
        # File 0 has a single event TTL, so the form renders no split checkbox for it.
        with pytest.raises(ValueError, match="nothing to split"):
            testing_api._drive_npm_configuration_form(
                template=npm_template_two_timestamp_columns,
                npm_timestamp_column_name=None,
                npm_time_unit=None,
                npm_split_events=[True, False],
            )

    def test_timestamp_column_on_single_column_session_raises(self, npm_template_single_timestamp_column):
        with pytest.raises(ValueError, match="only one timestamp column"):
            testing_api._drive_npm_configuration_form(
                template=npm_template_single_timestamp_column,
                npm_timestamp_column_name="Timestamp",
                npm_time_unit=None,
                npm_split_events=None,
            )


def test_step1_invalid_run_name_raises_from_the_page_alert(staged_csv_session):
    with pytest.raises(ValueError, match="forbidden character"):
        testing_api.step1(
            base_dir=staged_csv_session["base_dir"],
            selected_folders=[staged_csv_session["session"]],
            store_id_to_store_label={
                "Sample_Control_Channel": "control_region",
                "Sample_Signal_Channel": "signal_region",
            },
            run_name="bad/name",
        )
