"""Unit tests for the channel-centric, schema-driven NWB-metadata core."""

from pathlib import Path

import numpy as np
import pytest

from guppy.utils import nwb_metadata as m

EXAMPLE_METADATA = Path(__file__).resolve().parents[2] / "data" / "fiber_photometry_metadata_example.yaml"


@pytest.fixture
def channels() -> list[m.Channel]:
    # Canonical order: control then signal within a region.
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
        assert [(c.region, c.role, c.store_name) for c in channels] == [
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


class TestYamlIO:
    def test_dump_then_load_round_trip(self, tmp_path):
        metadata = {"NWBFile": {"lab": "Lerner"}, "Ophys": {"FiberPhotometry": {"OpticalFibers": [{"name": "f"}]}}}
        path = tmp_path / "meta.yaml"
        m.dump_yaml(metadata, path)
        assert m.load_yaml(path) == metadata

    def test_loads_blank_text_returns_empty_dict(self):
        assert m.loads_yaml("") == {}
