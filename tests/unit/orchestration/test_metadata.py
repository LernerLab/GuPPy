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

    def test_save_validation_errors_do_not_write(self, captured, tmp_path):
        path = str(tmp_path / "out" / METADATA_FILENAME)
        build_metadata_template("Photo (run1)", CHANNELS, {}, path)
        selector = captured[0]
        selector.set_yaml({})  # valid YAML but missing every required field

        selector.save.clicks += 1

        assert "Missing required metadata for NWB export" in selector.alert.object
        assert not (tmp_path / "out").exists()


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
        input_parameters = {"selected_runs": {str(session): ["run1"]}}

        # Must return without raising and without opening a server.
        orchestrate_metadata_page(input_parameters)
