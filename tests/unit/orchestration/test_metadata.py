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
    build_metadata_template,
    build_metadata_templates,
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
    return build_metadata_dict(
        devices=COMPLETE_DEVICES, channel_rows=COMPLETE_ROWS, scalars=COMPLETE_SCALARS, channels=CHANNELS
    )


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
        build_metadata_template(session_label="Photo (run1)", channels=CHANNELS, metadata={}, metadata_yaml_path=path)
        selector = captured[0]
        selector.set_from_metadata(_complete_metadata())

        selector.build_config.clicks += 1

        assert selector.get_yaml() != {}
        assert "No alerts" in selector.alert.object

    def test_build_config_incomplete_metadata_lists_missing(self, captured, tmp_path):
        path = str(tmp_path / "out" / METADATA_FILENAME)
        build_metadata_template(session_label="Photo (run1)", channels=CHANNELS, metadata={}, metadata_yaml_path=path)
        selector = captured[0]

        selector.build_config.clicks += 1

        assert "Missing required metadata for NWB export" in selector.alert.object

    def test_build_config_value_error_sets_alert_and_keeps_yaml(self, captured, tmp_path, monkeypatch):
        path = str(tmp_path / "out" / METADATA_FILENAME)
        build_metadata_template(session_label="Photo (run1)", channels=CHANNELS, metadata={}, metadata_yaml_path=path)
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
        build_metadata_template(session_label="Photo (run1)", channels=CHANNELS, metadata={}, metadata_yaml_path=path)
        selector = captured[0]
        built = _complete_metadata()
        selector.set_yaml(built)

        selector.save.clicks += 1

        assert load_yaml(path) == built
        assert selector.path.value == path
        assert "No alerts" in selector.alert.object

    def test_save_invalid_yaml_does_not_write(self, captured, tmp_path):
        path = str(tmp_path / "out" / METADATA_FILENAME)
        build_metadata_template(session_label="Photo (run1)", channels=CHANNELS, metadata={}, metadata_yaml_path=path)
        selector = captured[0]
        selector.code_editor.value = "*undefined_alias"

        selector.save.clicks += 1

        assert "Invalid YAML" in selector.alert.object
        assert not (tmp_path / "out").exists()

    def test_save_reports_a_missing_session_start_time_when_required(self, captured, tmp_path):
        # A format that records no start time makes the form its only source, so Save must refuse
        # metadata that would otherwise be complete.
        path = str(tmp_path / "out" / METADATA_FILENAME)
        build_metadata_template(
            session_label="Photo (run1)",
            channels=CHANNELS,
            metadata={},
            metadata_yaml_path=path,
            require_session_start_time=True,
        )
        selector = captured[0]
        selector.set_yaml(_complete_metadata())

        selector.save.clicks += 1

        assert "NWBFile.session_start_time is required" in selector.alert.object
        assert not (tmp_path / "out").exists()

    def test_save_writes_when_the_required_session_start_time_is_supplied(self, captured, tmp_path):
        path = str(tmp_path / "out" / METADATA_FILENAME)
        build_metadata_template(
            session_label="Photo (run1)",
            channels=CHANNELS,
            metadata={},
            metadata_yaml_path=path,
            require_session_start_time=True,
        )
        selector = captured[0]
        built = build_metadata_dict(
            devices=COMPLETE_DEVICES,
            channel_rows=COMPLETE_ROWS,
            scalars={**COMPLETE_SCALARS, "session_start_time": "2018-10-30T10:33:32-05:00"},
            channels=CHANNELS,
        )
        selector.set_yaml(built)

        selector.save.clicks += 1

        assert load_yaml(path) == built
        assert "No alerts" in selector.alert.object

    def test_save_validation_errors_do_not_write(self, captured, tmp_path):
        path = str(tmp_path / "out" / METADATA_FILENAME)
        build_metadata_template(session_label="Photo (run1)", channels=CHANNELS, metadata={}, metadata_yaml_path=path)
        selector = captured[0]
        selector.set_yaml({})  # valid YAML but missing every required field

        selector.save.clicks += 1

        assert "Missing required metadata for NWB export" in selector.alert.object
        assert not (tmp_path / "out").exists()


class TestRequiresSessionStartTime:
    """Only the formats whose raw files record a start time leave the field optional."""

    @pytest.fixture
    def required_flags(self, monkeypatch):
        """Record the ``require_session_start_time`` each built page was given."""
        flags = []
        monkeypatch.setattr(
            metadata_module,
            "build_metadata_template",
            lambda *args, require_session_start_time, **kwargs: flags.append(require_session_start_time),
        )
        return flags

    @pytest.fixture
    def session_path(self, tmp_path):
        session = tmp_path / "Photo_session"
        output_dir = session / "Photo_session_output_run1"
        output_dir.mkdir(parents=True)
        np.savetxt(
            output_dir / "storesList.csv",
            np.array([["Dv1A", "Dv2A"], ["control_dms", "signal_dms"]]),
            delimiter=",",
            fmt="%s",
        )
        return session

    @staticmethod
    def _input_parameters(session_path):
        return {"selected_runs": {str(session_path): ["run1"]}, "combine_data": False}

    def test_tdt_session_does_not_require_one(self, required_flags, session_path):
        (session_path / "Photo_session.tsq").write_bytes(b"\x00")

        build_metadata_templates(inputParameters=self._input_parameters(session_path))

        assert required_flags == [False]

    def test_csv_session_requires_one(self, required_flags, session_path):
        (session_path / "signal_dms.csv").write_text("timestamps,data,sampling_rate\n0.0,1.0,100.0\n")

        build_metadata_templates(inputParameters=self._input_parameters(session_path))

        assert required_flags == [True]

    def test_unresolvable_format_is_reported_rather_than_guessed(self, required_flags, session_path):
        # No acquisition files at all. Surfaced here, where the user can act on it, rather than
        # opening a form for a session the export will refuse.
        with pytest.raises(ValueError) as excinfo:
            orchestrate_metadata_page(self._input_parameters(session_path))

        assert "No acquisition data was found" in str(excinfo.value)
        assert required_flags == []


class TestOrchestrateMetadataPage:
    def test_builder_builds_pages_without_serving(self, panel_extension, tmp_path):
        session = tmp_path / "Photo_session"
        output_dir = session / "Photo_session_output_run1"
        output_dir.mkdir(parents=True)
        (session / "Photo_session.tsq").write_bytes(b"\x00")
        np.savetxt(
            output_dir / "storesList.csv",
            np.array([["Dv1A", "Dv2A"], ["control_dms", "signal_dms"]]),
            delimiter=",",
            fmt="%s",
        )
        input_parameters = {"selected_runs": {str(session): ["run1"]}, "combine_data": False}

        # Must return without raising and without opening a server.
        build_metadata_templates(inputParameters=input_parameters)

    def test_orchestrator_serves_each_built_page(self, panel_extension, monkeypatch, tmp_path):
        session = tmp_path / "Photo_session"
        output_dir = session / "Photo_session_output_run1"
        output_dir.mkdir(parents=True)
        (session / "Photo_session.tsq").write_bytes(b"\x00")
        np.savetxt(
            output_dir / "storesList.csv",
            np.array([["Dv1A", "Dv2A"], ["control_dms", "signal_dms"]]),
            delimiter=",",
            fmt="%s",
        )
        input_parameters = {"selected_runs": {str(session): ["run1"]}, "combine_data": False}

        served_ports = []
        monkeypatch.setattr(
            metadata_module.pn.template.BootstrapTemplate, "show", lambda self, port: served_ports.append(port)
        )

        orchestrate_metadata_page(input_parameters)

        assert len(served_ports) == 1

    def test_combined_run_is_refused_before_any_page_is_built(self, panel_extension, tmp_path):
        # Step 6 edits one metadata file per session output directory, which combining collapses,
        # so the form has nothing coherent to edit. Refused the same way Step 7 refuses it.
        input_parameters = {"selected_runs": {str(tmp_path / "Photo_session"): ["run1"]}, "combine_data": True}

        with pytest.raises(ValueError) as excinfo:
            orchestrate_metadata_page(input_parameters)

        assert "does not support combine_data=True" in str(excinfo.value)


class TestOrchestrateMetadataPageSkipsNwbSources:
    """A session GuPPy processed out of an NWB file gets no form: the export adds its outputs to
    the file it came from, which already carries everything the form collects."""

    @pytest.fixture
    def built_sessions(self, monkeypatch):
        """Record which sessions a page was built for, without building any Panel widgets."""
        labels = []
        monkeypatch.setattr(
            metadata_module,
            "build_metadata_template",
            lambda session_label, *args, **kwargs: labels.append(session_label),
        )
        return labels

    @pytest.fixture
    def raw_session(self, tmp_path):
        session = tmp_path / "Photo_raw"
        (session / "Photo_raw_output_run1").mkdir(parents=True)
        (session / "Photo_raw.tsq").write_bytes(b"\x00")
        np.savetxt(
            session / "Photo_raw_output_run1" / "storesList.csv",
            np.array([["Dv1A", "Dv2A"], ["control_dms", "signal_dms"]]),
            delimiter=",",
            fmt="%s",
        )
        return session

    @pytest.fixture
    def nwb_session(self, tmp_path):
        session = tmp_path / "Photo_nwb"
        (session / "Photo_nwb_output_run1").mkdir(parents=True)
        (session / "session.nwb").write_bytes(b"\x00")
        return session

    def test_nwb_session_is_skipped_and_the_raw_one_is_not(self, built_sessions, raw_session, nwb_session):
        input_parameters = {
            "selected_runs": {str(raw_session): ["run1"], str(nwb_session): ["run1"]},
            "combine_data": False,
        }

        build_metadata_templates(inputParameters=input_parameters)

        assert built_sessions == ["Photo_raw (run1)"]

    def test_an_all_nwb_batch_builds_nothing(self, built_sessions, nwb_session):
        input_parameters = {"selected_runs": {str(nwb_session): ["run1"]}, "combine_data": False}

        build_metadata_templates(inputParameters=input_parameters)

        assert built_sessions == []

    def test_a_dandi_batch_builds_nothing(self, built_sessions, tmp_path):
        # A DANDI session's folder holds only GuPPy's outputs, so there is nothing to detect in it.
        session = tmp_path / "Photo_dandi"
        (session / "Photo_dandi_output_run1").mkdir(parents=True)
        input_parameters = {
            "mode": "dandi",
            "dandi_uri_map": {str(session): "dandi://000971/sub-112/sub-112_ses-1.nwb"},
            "selected_runs": {str(session): ["run1"]},
            "combine_data": False,
        }

        build_metadata_templates(inputParameters=input_parameters)

        assert built_sessions == []
