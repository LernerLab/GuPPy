import types

import pytest

import guppy.testing.api as testing_api


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
def valid_storenames_map():
    return {"RawSignal": "signal_region"}


class TestSharedApiValidation:
    @pytest.mark.parametrize("step_name", ["step1", "step2", "step3", "step4", "step5"])
    def test_steps_require_existing_base_directory(self, api_workspace, valid_storenames_map, step_name):
        step = getattr(testing_api, step_name)
        kwargs = {
            "base_dir": api_workspace["missing_base_directory"],
            "selected_folders": [api_workspace["session_directory"]],
        }
        if step_name == "step1":
            kwargs["storenames_map"] = valid_storenames_map
        if step_name in ("step2", "step3", "step4", "step5"):
            kwargs["selected_runs"] = {api_workspace["session_directory"]: ["1"]}

        with pytest.raises(ValueError, match="base_dir does not exist or is not a directory"):
            step(**kwargs)

    @pytest.mark.parametrize("step_name", ["step1", "step2", "step3", "step4", "step5"])
    def test_steps_require_selected_folders(self, api_workspace, valid_storenames_map, step_name):
        step = getattr(testing_api, step_name)
        kwargs = {
            "base_dir": api_workspace["base_directory"],
            "selected_folders": [],
        }
        if step_name == "step1":
            kwargs["storenames_map"] = valid_storenames_map
        if step_name in ("step2", "step3", "step4", "step5"):
            kwargs["selected_runs"] = {}

        with pytest.raises(ValueError, match="selected_folders must be a non-empty iterable"):
            step(**kwargs)

    @pytest.mark.parametrize("step_name", ["step1", "step2", "step3", "step4", "step5"])
    def test_steps_reject_missing_session_directory(self, api_workspace, valid_storenames_map, step_name):
        step = getattr(testing_api, step_name)
        kwargs = {
            "base_dir": api_workspace["base_directory"],
            "selected_folders": [api_workspace["missing_session_directory"]],
        }
        if step_name == "step1":
            kwargs["storenames_map"] = valid_storenames_map
        if step_name in ("step2", "step3", "step4", "step5"):
            kwargs["selected_runs"] = {api_workspace["missing_session_directory"]: ["1"]}

        with pytest.raises(ValueError, match="Session path does not exist or is not a directory"):
            step(**kwargs)

    @pytest.mark.parametrize("step_name", ["step1", "step2", "step3", "step4", "step5"])
    def test_steps_require_sessions_under_base_directory(self, api_workspace, valid_storenames_map, step_name):
        step = getattr(testing_api, step_name)
        kwargs = {
            "base_dir": api_workspace["base_directory"],
            "selected_folders": [api_workspace["foreign_session_directory"]],
        }
        if step_name == "step1":
            kwargs["storenames_map"] = valid_storenames_map
        if step_name in ("step2", "step3", "step4", "step5"):
            kwargs["selected_runs"] = {api_workspace["foreign_session_directory"]: ["1"]}

        with pytest.raises(ValueError, match="must share the same parent equal to base_dir"):
            step(**kwargs)


class TestStep1Validation:
    @pytest.mark.parametrize(
        ("storenames_map", "message"),
        [
            ({}, "storenames_map must be a non-empty dict"),
            ({"": "signal_region"}, "Invalid storename key"),
            ({"RawSignal": ""}, "Invalid semantic name"),
            ({1: "signal_region"}, "Invalid storename key"),
            ({"RawSignal": None}, "Invalid semantic name"),
        ],
    )
    def test_step1_validates_storenames_map(self, api_workspace, storenames_map, message):
        with pytest.raises(ValueError, match=message):
            testing_api.step1(
                base_dir=api_workspace["base_directory"],
                selected_folders=[api_workspace["session_directory"]],
                storenames_map=storenames_map,
            )


class TestApiRuntimeErrors:
    @pytest.mark.parametrize("step_name", ["step1", "step2", "step3", "step4", "step5"])
    def test_steps_require_get_input_parameters_hook(self, api_workspace, valid_storenames_map, monkeypatch, step_name):
        monkeypatch.setattr(
            testing_api,
            "build_homepage",
            lambda: FakeTemplate(widgets={"files_1": types.SimpleNamespace(value=None)}),
        )

        step = getattr(testing_api, step_name)
        kwargs = {
            "base_dir": api_workspace["base_directory"],
            "selected_folders": [api_workspace["session_directory"]],
        }
        if step_name == "step1":
            kwargs["storenames_map"] = valid_storenames_map
        if step_name in ("step2", "step3", "step4", "step5"):
            kwargs["selected_runs"] = {api_workspace["session_directory"]: ["1"]}

        with pytest.raises(RuntimeError, match="getInputParameters"):
            step(**kwargs)

    @pytest.mark.parametrize("step_name", ["step1", "step2", "step3", "step4", "step5"])
    def test_steps_require_files_widget(self, api_workspace, valid_storenames_map, monkeypatch, step_name):
        monkeypatch.setattr(
            testing_api,
            "build_homepage",
            lambda: FakeTemplate(hooks={"getInputParameters": lambda: {}}),
        )

        step = getattr(testing_api, step_name)
        kwargs = {
            "base_dir": api_workspace["base_directory"],
            "selected_folders": [api_workspace["session_directory"]],
        }
        if step_name == "step1":
            kwargs["storenames_map"] = valid_storenames_map
        if step_name in ("step2", "step3", "step4", "step5"):
            kwargs["selected_runs"] = {api_workspace["session_directory"]: ["1"]}

        with pytest.raises(RuntimeError, match="files_1"):
            step(**kwargs)

    def test_save_parameters_requires_get_input_parameters_hook(self, api_workspace, monkeypatch):
        monkeypatch.setattr(
            testing_api,
            "build_homepage",
            lambda: FakeTemplate(widgets={"files_1": types.SimpleNamespace(value=None)}),
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
            lambda: FakeTemplate(hooks={"getInputParameters": lambda: {}}),
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


class TestNormalizeGroupSelectedRuns:
    def test_raises_when_runs_provided_but_no_group_folders(self):
        with pytest.raises(ValueError, match="no group_folders were selected"):
            testing_api._normalize_group_selected_runs({"/abs/session": ["run1"]}, [])

    def test_returns_empty_when_no_group_folders_and_no_runs(self):
        assert testing_api._normalize_group_selected_runs(None, []) == {}
        assert testing_api._normalize_group_selected_runs({}, []) == {}

    def test_raises_when_group_folders_present_but_runs_is_none(self, tmp_path):
        session = tmp_path / "session"
        session.mkdir()
        with pytest.raises(ValueError, match="group_selected_runs is required"):
            testing_api._normalize_group_selected_runs(None, [str(session)])

    def test_delegates_to_normalize_selected_runs_with_group_param_name(self, tmp_path):
        session = tmp_path / "session"
        session.mkdir()
        with pytest.raises(ValueError, match="group_selected_runs"):
            testing_api._normalize_group_selected_runs({str(session): []}, [str(session)])
