"""Unit tests for the Step 6 metadata orchestration (page builder + Save/Build callbacks).

``build_metadata_template`` wires the Build-preview and Save buttons to closures over a
``MetadataSelector``. The selector is driven in-process: values are set in Python and the
buttons are fired synchronously (``button.clicks += 1``), so the callbacks are exercised
without a browser.
"""

import numpy as np
import pytest

from guppy.orchestration import metadata as metadata_module
from guppy.orchestration.metadata import (
    METADATA_FILENAME,
    _requires_session_start_time,
    _selected_session_runs,
    build_metadata_template,
    orchestrate_metadata_page,
)
from guppy.utils.nwb_metadata import Channel, build_metadata_dict, load_yaml

CHANNELS = [Channel("dms", "control", "Dv1A"), Channel("dms", "signal", "Dv2A")]

COMPLETE_DEVICES = {
    "optical_fiber_model": [{"name": "fmodel", "numerical_aperture": 0.48, "manufacturer": "Doric"}],
    "optical_fiber": [{"name": "fiber", "model": "fmodel"}],
    "excitation_source_model": [
        {"name": "smodel", "source_type": "LED", "excitation_mode": "one-photon", "manufacturer": "Thorlabs"}
    ],
    "excitation_source": [{"name": "source", "model": "smodel"}],
    "photodetector_model": [{"name": "pmodel", "detector_type": "photodiode", "manufacturer": "Newport"}],
    "photodetector": [{"name": "detector", "model": "pmodel"}],
    "indicator": [{"name": "gcamp", "label": "GCaMP6f"}],
}
COMPLETE_ROWS = [
    {
        "excitation_wavelength_in_nm": 405.0,
        "emission_wavelength_in_nm": 525.0,
        "indicator": "gcamp",
        "optical_fiber": "fiber",
        "excitation_source": "source",
        "photodetector": "detector",
    },
    {
        "excitation_wavelength_in_nm": 465.0,
        "emission_wavelength_in_nm": 525.0,
        "indicator": "gcamp",
        "optical_fiber": "fiber",
        "excitation_source": "source",
        "photodetector": "detector",
    },
]
COMPLETE_SCALARS = {"session_description": "RI30", "subject_id": "63", "sex": "M", "species": "Mus musculus"}


def _complete_metadata() -> dict:
    return build_metadata_dict(COMPLETE_DEVICES, COMPLETE_ROWS, COMPLETE_SCALARS, CHANNELS)


class TestSelectedSessionRuns:
    def test_flattens_sessions_and_runs_in_order(self):
        input_parameters = {"selected_runs": {"/data/A": ["run1", "run2"], "/data/B": ["run1"]}}
        assert _selected_session_runs(input_parameters) == [
            ("/data/A", "run1"),
            ("/data/A", "run2"),
            ("/data/B", "run1"),
        ]


class TestBuildMetadataTemplate:
    @pytest.fixture
    def captured(self, panel_extension, monkeypatch):
        """Capture the MetadataSelector built inside build_metadata_template so tests can drive it."""
        instances = []
        real_selector = metadata_module.MetadataSelector

        def capture(*args, **kwargs):
            selector = real_selector(*args, **kwargs)
            instances.append(selector)
            return selector

        monkeypatch.setattr(metadata_module, "MetadataSelector", capture)
        return instances

    def test_build_config_complete_metadata_clears_alerts(self, captured, tmp_path):
        path = str(tmp_path / "out" / METADATA_FILENAME)
        build_metadata_template("Photo (run1)", CHANNELS, {}, path)
        selector = captured[0]
        selector.set_from_metadata(_complete_metadata())

        selector.build_config.clicks += 1

        assert selector.get_yaml() != {}
        assert "No alerts" in selector.alert.object

    def test_build_config_incomplete_metadata_lists_missing(self, captured, tmp_path):
        path = str(tmp_path / "out" / METADATA_FILENAME)
        build_metadata_template("Photo (run1)", CHANNELS, {}, path)
        selector = captured[0]

        selector.build_config.clicks += 1

        assert "Missing required metadata for NWB export" in selector.alert.object

    def test_build_config_value_error_sets_alert_and_keeps_yaml(self, captured, tmp_path, monkeypatch):
        path = str(tmp_path / "out" / METADATA_FILENAME)
        build_metadata_template("Photo (run1)", CHANNELS, {}, path)
        selector = captured[0]
        selector.set_yaml({"sentinel": True})

        def _raise(*args, **kwargs):
            raise ValueError("bad device combination")

        monkeypatch.setattr(metadata_module, "build_metadata_dict", _raise)
        selector.build_config.clicks += 1

        assert "bad device combination" in selector.alert.object
        # The previewed YAML must be left untouched when the build fails.
        assert selector.get_yaml() == {"sentinel": True}

    def test_save_writes_complete_metadata(self, captured, tmp_path):
        path = str(tmp_path / "out" / METADATA_FILENAME)
        build_metadata_template("Photo (run1)", CHANNELS, {}, path)
        selector = captured[0]
        built = _complete_metadata()
        selector.set_yaml(built)

        selector.save.clicks += 1

        assert load_yaml(path) == built
        assert selector.path.value == path
        assert "No alerts" in selector.alert.object

    def test_save_invalid_yaml_does_not_write(self, captured, tmp_path):
        path = str(tmp_path / "out" / METADATA_FILENAME)
        build_metadata_template("Photo (run1)", CHANNELS, {}, path)
        selector = captured[0]
        selector.code_editor.value = "*undefined_alias"

        selector.save.clicks += 1

        assert "Invalid YAML" in selector.alert.object
        assert not (tmp_path / "out").exists()

    def test_save_reports_a_missing_session_start_time_when_required(self, captured, tmp_path):
        # A format that records no start time makes the form its only source, so Save must refuse
        # metadata that would otherwise be complete.
        path = str(tmp_path / "out" / METADATA_FILENAME)
        build_metadata_template("Photo (run1)", CHANNELS, {}, path, require_session_start_time=True)
        selector = captured[0]
        selector.set_yaml(_complete_metadata())

        selector.save.clicks += 1

        assert "NWBFile.session_start_time is required" in selector.alert.object
        assert not (tmp_path / "out").exists()

    def test_save_writes_when_the_required_session_start_time_is_supplied(self, captured, tmp_path):
        path = str(tmp_path / "out" / METADATA_FILENAME)
        build_metadata_template("Photo (run1)", CHANNELS, {}, path, require_session_start_time=True)
        selector = captured[0]
        built = build_metadata_dict(
            COMPLETE_DEVICES,
            COMPLETE_ROWS,
            {**COMPLETE_SCALARS, "session_start_time": "2018-10-30T10:33:32-05:00"},
            CHANNELS,
        )
        selector.set_yaml(built)

        selector.save.clicks += 1

        assert load_yaml(path) == built
        assert "No alerts" in selector.alert.object

    def test_save_validation_errors_do_not_write(self, captured, tmp_path):
        path = str(tmp_path / "out" / METADATA_FILENAME)
        build_metadata_template("Photo (run1)", CHANNELS, {}, path)
        selector = captured[0]
        selector.set_yaml({})  # valid YAML but missing every required field

        selector.save.clicks += 1

        assert "Missing required metadata for NWB export" in selector.alert.object
        assert not (tmp_path / "out").exists()


class TestRequiresSessionStartTime:
    @pytest.fixture
    def session_path(self, tmp_path):
        session = tmp_path / "Photo_session"
        session.mkdir()
        return session

    def test_tdt_session_does_not_require_one(self, session_path):
        (session_path / "Photo_session.tsq").write_bytes(b"\x00")
        assert _requires_session_start_time(str(session_path)) is False

    def test_csv_session_requires_one(self, session_path):
        (session_path / "signal_dms.csv").write_text("timestamps,data,sampling_rate\n0.0,1.0,100.0\n")
        assert _requires_session_start_time(str(session_path)) is True

    def test_unresolvable_format_requires_one(self, session_path):
        # Step 7 reports the format problem itself; asking for the start time here costs nothing.
        assert _requires_session_start_time(str(session_path)) is True


class TestOrchestrateMetadataPage:
    def test_headless_builds_pages_without_serving(self, panel_extension, tmp_path):
        # GUPPY_BASE_DIR is set by the test conftest -> headless: pages are built but never served.
        session = tmp_path / "Photo_session"
        output_dir = session / "Photo_session_output_run1"
        output_dir.mkdir(parents=True)
        np.savetxt(
            output_dir / "storesList.csv",
            np.array([["Dv1A", "Dv2A"], ["control_dms", "signal_dms"]]),
            delimiter=",",
            fmt="%s",
        )
        input_parameters = {"selected_runs": {str(session): ["run1"]}, "combine_data": False}

        # Must return without raising and without opening a server.
        orchestrate_metadata_page(input_parameters)

    def test_combined_run_is_refused_before_any_page_is_built(self, panel_extension, tmp_path):
        # Step 6 edits one metadata file per session output directory, which combining collapses,
        # so the form has nothing coherent to edit. Refused the same way Step 7 refuses it.
        input_parameters = {"selected_runs": {str(tmp_path / "Photo_session"): ["run1"]}, "combine_data": True}

        with pytest.raises(ValueError) as excinfo:
            orchestrate_metadata_page(input_parameters)

        assert "does not support combine_data=True" in str(excinfo.value)
