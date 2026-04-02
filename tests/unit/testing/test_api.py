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
    @pytest.mark.parametrize("step_name", ["step2", "step3", "step4", "step5", "step6"])
    def test_steps_require_existing_base_directory(self, api_workspace, valid_storenames_map, step_name):
        step = getattr(testing_api, step_name)
        kwargs = {
            "base_dir": api_workspace["missing_base_directory"],
            "selected_folders": [api_workspace["session_directory"]],
        }
        if step_name == "step2":
            kwargs["storenames_map"] = valid_storenames_map

        with pytest.raises(ValueError, match="base_dir does not exist or is not a directory"):
            step(**kwargs)

    @pytest.mark.parametrize("step_name", ["step2", "step3", "step4", "step5", "step6"])
    def test_steps_require_selected_folders(self, api_workspace, valid_storenames_map, step_name):
        step = getattr(testing_api, step_name)
        kwargs = {
            "base_dir": api_workspace["base_directory"],
            "selected_folders": [],
        }
        if step_name == "step2":
            kwargs["storenames_map"] = valid_storenames_map

        with pytest.raises(ValueError, match="selected_folders must be a non-empty iterable"):
            step(**kwargs)

    @pytest.mark.parametrize("step_name", ["step2", "step3", "step4", "step5", "step6"])
    def test_steps_reject_missing_session_directory(self, api_workspace, valid_storenames_map, step_name):
        step = getattr(testing_api, step_name)
        kwargs = {
            "base_dir": api_workspace["base_directory"],
            "selected_folders": [api_workspace["missing_session_directory"]],
        }
        if step_name == "step2":
            kwargs["storenames_map"] = valid_storenames_map

        with pytest.raises(ValueError, match="Session path does not exist or is not a directory"):
            step(**kwargs)

    @pytest.mark.parametrize("step_name", ["step2", "step3", "step4", "step5", "step6"])
    def test_steps_require_sessions_under_base_directory(self, api_workspace, valid_storenames_map, step_name):
        step = getattr(testing_api, step_name)
        kwargs = {
            "base_dir": api_workspace["base_directory"],
            "selected_folders": [api_workspace["foreign_session_directory"]],
        }
        if step_name == "step2":
            kwargs["storenames_map"] = valid_storenames_map

        with pytest.raises(ValueError, match="must share the same parent equal to base_dir"):
            step(**kwargs)


class TestStep2Validation:
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
    def test_step2_validates_storenames_map(self, api_workspace, storenames_map, message):
        with pytest.raises(ValueError, match=message):
            testing_api.step2(
                base_dir=api_workspace["base_directory"],
                selected_folders=[api_workspace["session_directory"]],
                storenames_map=storenames_map,
            )


class TestApiRuntimeErrors:
    @pytest.mark.parametrize("step_name", ["step2", "step3", "step4", "step5", "step6"])
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
        if step_name == "step2":
            kwargs["storenames_map"] = valid_storenames_map

        with pytest.raises(RuntimeError, match="getInputParameters"):
            step(**kwargs)

    @pytest.mark.parametrize("step_name", ["step2", "step3", "step4", "step5", "step6"])
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
        if step_name == "step2":
            kwargs["storenames_map"] = valid_storenames_map

        with pytest.raises(RuntimeError, match="files_1"):
            step(**kwargs)

    def test_step1_requires_onclick_process_hook(self, api_workspace, monkeypatch):
        monkeypatch.setattr(
            testing_api,
            "build_homepage",
            lambda: FakeTemplate(widgets={"files_1": types.SimpleNamespace(value=None)}),
        )

        with pytest.raises(RuntimeError, match="onclickProcess"):
            testing_api.step1(
                base_dir=api_workspace["base_directory"],
                selected_folders=[api_workspace["session_directory"]],
            )

    def test_step1_requires_files_widget(self, api_workspace, monkeypatch):
        monkeypatch.setattr(
            testing_api,
            "build_homepage",
            lambda: FakeTemplate(hooks={"onclickProcess": lambda: None}),
        )

        with pytest.raises(RuntimeError, match="files_1"):
            testing_api.step1(
                base_dir=api_workspace["base_directory"],
                selected_folders=[api_workspace["session_directory"]],
            )
