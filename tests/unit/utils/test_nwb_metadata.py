"""Unit tests for the channel-centric, schema-driven NWB-metadata core."""

from pathlib import Path

import numpy as np
import pytest

from guppy.utils import nwb_metadata as m

EXAMPLE_METADATA = Path(__file__).resolve().parents[2] / "data" / "fiber_photometry_metadata_example.yaml"


@pytest.fixture
def channels() -> list[m.Channel]:
    # Canonical order: control then signal within a recording site.
    return [m.Channel("dms", "control", "Dv1A"), m.Channel("dms", "signal", "Dv2A")]


class TestDeriveChannels:
    def test_parses_storeslist_into_ordered_channels(self, tmp_path):
        np.savetxt(
            tmp_path / "storesList.csv",
            np.array([["Dv1A", "Dv2A", "PrtN"], ["control_dms", "signal_dms", "port_entries_dms"]]),
            delimiter=",",
            fmt="%s",
        )
        channels = m.derive_channels(str(tmp_path))
        assert [(c.recording_site, c.role, c.store_name) for c in channels] == [
            ("dms", "control", "Dv1A"),
            ("dms", "signal", "Dv2A"),
        ]
        assert channels[0].response_series_name == "dms_isosbestic_control"
        assert channels[1].response_series_name == "dms_calcium_signal"


class TestFieldSpecs:
    def test_model_and_instance_are_separate_categories(self):
        model = {s.name: s for s in m.field_specs("optical_fiber_model")}
        instance = {s.name: s for s in m.field_specs("optical_fiber")}
        # The model holds the manufacturer specs (numerical_aperture required).
        assert model["numerical_aperture"].required is True
        assert model["core_diameter_in_um"].required is False
        assert "numerical_aperture" not in instance  # not duplicated onto the instance
        # The instance links to a model and carries the (mandatory) fiber insertion group.
        assert instance["model"].link_target == "optical_fiber_model"
        assert instance["depth_in_mm"].target == "insertion"
        # Deprecated instance dupes are dropped.
        assert "model_name" not in instance and "manufacturer" not in instance

    def test_enumerated_fields_have_options(self):
        source = {s.name: s for s in m.field_specs("excitation_source_model")}
        assert source["source_type"].options == ("LED", "Gas Laser", "Solid-State Laser")
        detector = {s.name: s for s in m.field_specs("photodetector_model")}
        assert "PMT" in detector["detector_type"].options

    def test_dependency_order_and_links(self):
        keys = list(m.CATEGORIES)
        # model before its instance; virus before injection before indicator.
        assert keys.index("optical_fiber_model") < keys.index("optical_fiber")
        assert keys.index("virus") < keys.index("virus_injection") < keys.index("indicator")
        assert {s.name: s for s in m.field_specs("indicator")}[
            "viral_vector_injection"
        ].link_target == "virus_injection"


class TestBuildMetadata:
    def test_instances_reuse_one_model_and_always_emit_insertion(self, channels):
        devices = {
            "optical_fiber_model": [{"name": "fmodel", "numerical_aperture": 0.48, "manufacturer": "Doric"}],
            "optical_fiber": [
                {"name": "dms_fiber", "model": "fmodel", "serial_number": "OF1", "depth_in_mm": 2.8},
                {"name": "dls_fiber", "model": "fmodel"},  # reuses the same model; no coordinates
            ],
        }
        fiber_photometry = m.build_metadata_dict(devices, [{}, {}], {}, channels)["Ophys"]["FiberPhotometry"]
        assert fiber_photometry["OpticalFiberModels"] == [
            {"name": "fmodel", "numerical_aperture": 0.48, "manufacturer": "Doric"}
        ]
        # Both instances link the one model; both carry the mandatory fiber_insertion group (2nd is empty).
        assert fiber_photometry["OpticalFibers"] == [
            {"name": "dms_fiber", "model": "fmodel", "serial_number": "OF1", "fiber_insertion": {"depth_in_mm": 2.8}},
            {"name": "dls_fiber", "model": "fmodel", "fiber_insertion": {}},
        ]

    def test_empty_channels_and_no_devices_yields_empty_metadata(self):
        # No channels -> no table/response series; no devices/scalars -> no Ophys/NWBFile/Subject keys.
        assert m.build_metadata_dict({}, [], {}, []) == {}

    def test_generates_table_rows_and_response_series_from_channels(self, channels):
        channel_rows = [
            {"excitation_wavelength_in_nm": 405.0, "emission_wavelength_in_nm": 525.0, "optical_fiber": "dms_fiber"},
            {"excitation_wavelength_in_nm": 465.0, "emission_wavelength_in_nm": 525.0, "optical_fiber": "dms_fiber"},
        ]
        metadata = m.build_metadata_dict({}, channel_rows, {}, channels)
        fiber_photometry = metadata["Ophys"]["FiberPhotometry"]
        rows = fiber_photometry["FiberPhotometryTable"]["rows"]
        assert rows[0] == {
            "name": 0,
            "location": "dms",
            "excitation_wavelength_in_nm": 405.0,
            "emission_wavelength_in_nm": 525.0,
            "optical_fiber": "dms_fiber",
        }
        series = fiber_photometry["FiberPhotometryResponseSeries"]
        # control channel -> Dv1A -> table row 0; signal -> Dv2A -> row 1.
        assert [(s["name"], s["stream_name"], s["fiber_photometry_table_region"]) for s in series] == [
            ("dms_isosbestic_control", "Dv1A", [0]),
            ("dms_calcium_signal", "Dv2A", [1]),
        ]

    def test_nwbfile_and_subject_assembly_drops_empty(self, channels):
        scalars = {
            "session_description": "RI30",
            "identifier": "id1",
            "lab": "",
            "experimenter": ["Doe, Jane"],
            "subject_id": "63",
            "sex": "M",
            "age": "P90D",
            "date_of_birth": "",
            "species": "Mus musculus",
        }
        metadata = m.build_metadata_dict({}, [{}, {}], scalars, channels)
        assert metadata["NWBFile"] == {
            "session_description": "RI30",
            "identifier": "id1",
            "experimenter": ["Doe, Jane"],
        }
        assert metadata["Subject"] == {"subject_id": "63", "sex": "M", "age": "P90D", "species": "Mus musculus"}


class TestParseMetadata:
    def test_splits_example_into_model_and_instance_categories(self, channels):
        example = m.load_yaml(EXAMPLE_METADATA)
        devices, channel_rows, scalars = m.parse_metadata_dict(example, channels)
        # Model and instance land in separate categories; the instance links the model + flattens insertion.
        assert devices["optical_fiber_model"][0]["name"] == "optical_fiber_model"
        assert devices["optical_fiber_model"][0]["numerical_aperture"] == 0.48
        fiber = devices["optical_fiber"][0]
        assert fiber["name"] == "optical_fiber"
        assert fiber["model"] == "optical_fiber_model"
        assert fiber["depth_in_mm"] == 2.8
        assert [d["name"] for d in devices["indicator"]] == ["dms_green_fluorophore", "dls_green_fluorophore"]

    def test_round_trip_build_parse_build(self, channels):
        devices = {
            "optical_fiber_model": [{"name": "fmodel", "numerical_aperture": 0.48, "manufacturer": "Doric"}],
            "optical_fiber": [{"name": "f", "model": "fmodel", "depth_in_mm": 2.8}],
        }
        rows = [{"excitation_wavelength_in_nm": 405.0}, {"excitation_wavelength_in_nm": 465.0}]
        built = m.build_metadata_dict(devices, rows, {}, channels)
        devices2, rows2, _ = m.parse_metadata_dict(built, channels)
        rebuilt = m.build_metadata_dict(devices2, rows2, {}, channels)
        assert (
            rebuilt["Ophys"]["FiberPhotometry"]["OpticalFibers"] == built["Ophys"]["FiberPhotometry"]["OpticalFibers"]
        )
        assert (
            rebuilt["Ophys"]["FiberPhotometry"]["OpticalFiberModels"]
            == built["Ophys"]["FiberPhotometry"]["OpticalFiberModels"]
        )


class TestValidateMetadata:
    def _complete_metadata(self, channels) -> dict:
        devices = {
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
        rows = [
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
        scalars = {"session_description": "RI30", "subject_id": "63", "sex": "M", "species": "Mus musculus"}
        return m.build_metadata_dict(devices, rows, scalars, channels)

    def test_complete_metadata_has_no_errors(self, channels):
        assert m.validate_metadata_dict(self._complete_metadata(channels), channels) == []

    def test_unlinked_optical_fiber_model_is_reported(self, channels):
        metadata = self._complete_metadata(channels)
        # Drop the instance's link to its model -> required-link violation.
        metadata["Ophys"]["FiberPhotometry"]["OpticalFibers"][0].pop("model")
        errors = m.validate_metadata_dict(metadata, channels)
        assert any("optical fiber 'fiber'" in error and "model is required" in error for error in errors)

    def test_dangling_link_and_missing_scalar_and_wavelength(self, channels):
        metadata = self._complete_metadata(channels)
        metadata["Subject"].pop("species")
        metadata["Ophys"]["FiberPhotometry"]["FiberPhotometryTable"]["rows"][0].pop("excitation_wavelength_in_nm")
        metadata["Ophys"]["FiberPhotometry"]["FiberPhotometryTable"]["rows"][1]["optical_fiber"] = "ghost"
        errors = m.validate_metadata_dict(metadata, channels)
        assert any("Subject.species is required" in error for error in errors)
        assert any("excitation_wavelength_in_nm is required" in error for error in errors)
        assert any("'ghost' is not a defined optical fiber" in error for error in errors)

    def test_missing_nwbfile_session_description_is_reported(self, channels):
        metadata = self._complete_metadata(channels)
        metadata["NWBFile"].pop("session_description")
        errors = m.validate_metadata_dict(metadata, channels)
        assert any("NWBFile.session_description is required" in error for error in errors)

    def test_missing_required_device_field_is_reported(self, channels):
        metadata = self._complete_metadata(channels)
        # numerical_aperture is a required (non-link) field on the optical fiber model.
        metadata["Ophys"]["FiberPhotometry"]["OpticalFiberModels"][0].pop("numerical_aperture")
        errors = m.validate_metadata_dict(metadata, channels)
        assert any("numerical_aperture is required" in error for error in errors)

    def test_dangling_device_link_is_reported(self, channels):
        metadata = self._complete_metadata(channels)
        metadata["Ophys"]["FiberPhotometry"]["OpticalFibers"][0]["model"] = "ghost"
        errors = m.validate_metadata_dict(metadata, channels)
        assert any("'ghost' is not a defined optical fiber model" in error for error in errors)

    def test_missing_required_channel_link_is_reported(self, channels):
        metadata = self._complete_metadata(channels)
        metadata["Ophys"]["FiberPhotometry"]["FiberPhotometryTable"]["rows"][0].pop("indicator")
        errors = m.validate_metadata_dict(metadata, channels)
        assert any("indicator link is required" in error for error in errors)


class TestIsEmpty:
    def test_none_and_nan_are_empty(self):
        assert m._is_empty(None) is True
        assert m._is_empty(np.nan) is True

    def test_blank_string_and_empty_collections_are_empty(self):
        assert m._is_empty("   ") is True
        assert m._is_empty([]) is True
        assert m._is_empty({}) is True

    def test_meaningful_values_are_not_empty(self):
        assert m._is_empty("x") is False
        assert m._is_empty(0.0) is False
        assert m._is_empty([1]) is False

    def test_drop_empty_filters_empty_values(self):
        assert m._drop_empty({"a": "x", "b": "", "c": None, "d": []}) == {"a": "x"}


class TestTypeIntrospection:
    def test_unknown_type_raises_assertion_error(self):
        with pytest.raises(AssertionError, match="not found in installed ndx namespaces"):
            m._type_spec("ThisTypeDoesNotExist")


class TestYamlIO:
    def test_dump_then_load_round_trip(self, tmp_path):
        metadata = {"NWBFile": {"lab": "Lerner"}, "Ophys": {"FiberPhotometry": {"OpticalFibers": [{"name": "f"}]}}}
        path = tmp_path / "meta.yaml"
        m.dump_yaml(metadata, path)
        assert m.load_yaml(path) == metadata

    def test_loads_blank_text_returns_empty_dict(self):
        assert m.loads_yaml("") == {}
