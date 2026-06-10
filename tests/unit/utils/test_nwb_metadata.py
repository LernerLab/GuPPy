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
    def test_optical_fiber_fields_required_and_targets(self):
        specs = {s.name: s for s in m.field_specs("optical_fiber")}
        assert specs["numerical_aperture"].required is True  # from the installed model spec
        assert specs["core_diameter_in_um"].required is False
        assert specs["numerical_aperture"].target == "model"
        assert specs["serial_number"].target == "instance"
        # fiber insertion coordinates are exposed and optional
        assert "depth_in_mm" in specs and specs["depth_in_mm"].target == "insertion"
        # deprecated instance dupes are dropped
        assert "model_name" not in specs

    def test_indicator_link_field(self):
        specs = {s.name: s for s in m.field_specs("indicator")}
        assert specs["label"].required is True
        assert specs["viral_vector_injection"].link_target == "virus_injection"


class TestBuildMetadata:
    def test_merged_device_splits_into_model_and_instance(self, channels):
        devices = {
            "optical_fiber": [
                {
                    "name": "dms_fiber",
                    "description": "a fiber",
                    "manufacturer": "Doric",
                    "numerical_aperture": 0.48,
                    "core_diameter_in_um": 400.0,
                    "serial_number": "OF1",
                    "depth_in_mm": 2.8,
                    "hemisphere": "right",
                }
            ]
        }
        metadata = m.build_metadata_dict(devices, [{}, {}], {}, channels)
        fiber_photometry = metadata["Ophys"]["FiberPhotometry"]
        assert fiber_photometry["OpticalFiberModels"] == [
            {
                "name": "dms_fiber_model",
                "numerical_aperture": 0.48,
                "core_diameter_in_um": 400.0,
                "manufacturer": "Doric",
            }
        ]
        assert fiber_photometry["OpticalFibers"] == [
            {
                "name": "dms_fiber",
                "model": "dms_fiber_model",
                "description": "a fiber",
                "serial_number": "OF1",
                "fiber_insertion": {"depth_in_mm": 2.8, "hemisphere": "right"},
            }
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
    def test_recombines_example_model_and_instance(self, channels):
        example = m.load_yaml(EXAMPLE_METADATA)
        devices, channel_rows, scalars = m.parse_metadata_dict(example, channels)
        fiber = devices["optical_fiber"][0]
        # name from the instance, numerical_aperture pulled back from the linked model, coords from fiber_insertion
        assert fiber["name"] == "optical_fiber"
        assert fiber["numerical_aperture"] == 0.48
        assert fiber["depth_in_mm"] == 2.8
        assert [d["name"] for d in devices["indicator"]] == ["dms_green_fluorophore", "dls_green_fluorophore"]

    def test_round_trip_build_parse_build(self, channels):
        devices = {"optical_fiber": [{"name": "f", "numerical_aperture": 0.48, "manufacturer": "Doric"}]}
        rows = [{"excitation_wavelength_in_nm": 405.0}, {"excitation_wavelength_in_nm": 465.0}]
        built = m.build_metadata_dict(devices, rows, {}, channels)
        devices2, rows2, _ = m.parse_metadata_dict(built, channels)
        rebuilt = m.build_metadata_dict(devices2, rows2, {}, channels)
        assert (
            rebuilt["Ophys"]["FiberPhotometry"]["OpticalFibers"] == built["Ophys"]["FiberPhotometry"]["OpticalFibers"]
        )


class TestYamlIO:
    def test_dump_then_load_round_trip(self, tmp_path):
        metadata = {"NWBFile": {"lab": "Lerner"}, "Ophys": {"FiberPhotometry": {"OpticalFibers": [{"name": "f"}]}}}
        path = tmp_path / "meta.yaml"
        m.dump_yaml(metadata, path)
        assert m.load_yaml(path) == metadata

    def test_loads_blank_text_returns_empty_dict(self):
        assert m.loads_yaml("") == {}
