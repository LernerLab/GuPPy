"""Unit tests for the pure NWB-export metadata helpers."""

from pathlib import Path

import pandas as pd
import pytest

from guppy.utils import nwb_metadata as m

EXAMPLE_METADATA = Path(__file__).resolve().parents[2] / "data" / "fiber_photometry_metadata_example.yaml"


@pytest.fixture
def example_metadata() -> dict:
    return m.load_yaml(EXAMPLE_METADATA)


class TestFlattenUnflatten:
    def test_flatten_nested_and_list_columns(self):
        schema = m.COMPONENT_SCHEMAS["OpticalFibers"]
        record = {
            "name": "optical_fiber",
            "model": "optical_fiber_model",
            "fiber_insertion": {"depth_in_mm": 2.8, "hemisphere": "right"},
        }
        row = m.flatten_record(record, schema)
        # Nested values are read out under dotted column names.
        assert row["fiber_insertion.depth_in_mm"] == 2.8
        assert row["fiber_insertion.hemisphere"] == "right"
        # Missing fields become empty strings, not KeyErrors.
        assert row["serial_number"] == ""

    def test_flatten_list_column_becomes_json_string(self):
        schema = m.COMPONENT_SCHEMAS["ExcitationSourceModels"]
        record = {"name": "src", "wavelength_range_in_nm": [400, 470]}
        row = m.flatten_record(record, schema)
        assert row["wavelength_range_in_nm"] == "[400, 470]"

    def test_unflatten_renests_and_parses_and_drops_empty(self):
        schema = m.COMPONENT_SCHEMAS["OpticalFibers"]
        row = {
            "name": "optical_fiber",
            "description": "",  # empty -> dropped
            "model": "optical_fiber_model",
            "serial_number": None,  # empty -> dropped
            "fiber_insertion.depth_in_mm": 2.8,
            "fiber_insertion.hemisphere": "right",
            "fiber_insertion.position_reference": "",  # empty -> dropped
        }
        record = m.unflatten_record(row, schema)
        assert record == {
            "name": "optical_fiber",
            "model": "optical_fiber_model",
            "fiber_insertion": {"depth_in_mm": 2.8, "hemisphere": "right"},
        }

    def test_unflatten_parses_json_list(self):
        schema = m.COMPONENT_SCHEMAS["FiberPhotometryResponseSeries"]
        row = {"name": "dms", "stream_name": "Dv2A", "fiber_photometry_table_region": "[0]"}
        record = m.unflatten_record(row, schema)
        assert record["fiber_photometry_table_region"] == [0]

    def test_unflatten_raises_on_malformed_list(self):
        schema = m.COMPONENT_SCHEMAS["ExcitationSourceModels"]
        row = {"name": "src", "wavelength_range_in_nm": "400, 470"}  # not JSON
        with pytest.raises(ValueError, match="must be a JSON list"):
            m.unflatten_record(row, schema)


class TestDataframeRoundTrip:
    def test_records_to_dataframe_columns_match_schema(self):
        schema = m.COMPONENT_SCHEMAS["Photodetectors"]
        dataframe = m.records_to_dataframe([{"name": "pd", "model": "pd_model"}], schema)
        assert list(dataframe.columns) == list(schema.columns)
        assert dataframe.iloc[0]["name"] == "pd"

    def test_dataframe_to_records_drops_fully_empty_rows(self):
        schema = m.COMPONENT_SCHEMAS["Photodetectors"]
        dataframe = pd.DataFrame(
            [
                {"name": "pd", "description": "", "model": "pd_model", "serial_number": ""},
                {"name": "", "description": "", "model": "", "serial_number": ""},  # trailing blank
            ],
            columns=list(schema.columns),
        )
        records = m.dataframe_to_records(dataframe, schema)
        assert records == [{"name": "pd", "model": "pd_model"}]

    def test_table_rows_round_trip(self):
        schema = m.COMPONENT_SCHEMAS["FiberPhotometryTable"]
        records = [{"name": 0, "location": "DMS", "indicator": "dms_green_fluorophore"}]
        dataframe = m.records_to_dataframe(records, schema)
        assert m.dataframe_to_records(dataframe, schema) == records


class TestMetadataAssembly:
    def test_build_metadata_dict_assembles_all_blocks(self):
        component_dataframes = {key: m.records_to_dataframe([], schema) for key, schema in m.COMPONENT_SCHEMAS.items()}
        component_dataframes["OpticalFibers"] = m.records_to_dataframe(
            [{"name": "optical_fiber", "fiber_insertion": {"depth_in_mm": 2.8}}],
            m.COMPONENT_SCHEMAS["OpticalFibers"],
        )
        scalars = {
            "session_description": "RI30 session",
            "identifier": "Photo_63_207",
            "lab": "Lerner Lab",
            "institution": "",
            "experimenter": ["Doe, Jane"],
            "subject_id": "63_207",
            "sex": "M",
            "age": "P90D",
            "date_of_birth": "",
            "species": "Mus musculus",
            "genotype": "",
            "strain": "",
            "fiber_photometry_table_name": "fiber_photometry_table",
            "fiber_photometry_table_description": "desc",
        }
        metadata = m.build_metadata_dict(component_dataframes, scalars)
        # Session and hardware live in the same self-contained dict.
        assert metadata["NWBFile"] == {
            "session_description": "RI30 session",
            "identifier": "Photo_63_207",
            "lab": "Lerner Lab",
            "experimenter": ["Doe, Jane"],
        }
        assert metadata["Subject"] == {"subject_id": "63_207", "sex": "M", "age": "P90D", "species": "Mus musculus"}
        assert metadata["Ophys"]["FiberPhotometry"]["OpticalFibers"] == [
            {"name": "optical_fiber", "fiber_insertion": {"depth_in_mm": 2.8}}
        ]
        # Empty component lists are omitted entirely.
        assert "Photodetectors" not in metadata["Ophys"]["FiberPhotometry"]

    def test_round_trip_preserves_example_components(self, example_metadata):
        dataframes, scalars = m.parse_metadata_dict(example_metadata)
        rebuilt = m.build_metadata_dict(dataframes, scalars)
        original_fp = example_metadata["Ophys"]["FiberPhotometry"]
        rebuilt_fp = rebuilt["Ophys"]["FiberPhotometry"]
        # Nested records, list fields, and the table all survive the round-trip.
        assert rebuilt_fp["OpticalFibers"] == original_fp["OpticalFibers"]
        assert rebuilt_fp["DichroicMirrorModels"] == original_fp["DichroicMirrorModels"]
        assert rebuilt_fp["FiberPhotometryTable"] == original_fp["FiberPhotometryTable"]

    def test_parse_metadata_dict_recovers_table_scalars(self, example_metadata):
        _, scalars = m.parse_metadata_dict(example_metadata)
        assert scalars["fiber_photometry_table_name"] == "fiber_photometry_table"

    def test_empty_dict_yields_structured_empty_tables(self):
        # A fresh session (no saved YAML) still renders every component table.
        dataframes, scalars = m.parse_metadata_dict({})
        assert set(dataframes) == set(m.COMPONENT_SCHEMAS)
        assert all(len(dataframe) == 0 for dataframe in dataframes.values())
        assert list(dataframes["OpticalFibers"].columns) == list(m.COMPONENT_SCHEMAS["OpticalFibers"].columns)
        assert scalars["session_description"] == ""


class TestYamlIO:
    def test_dump_then_load_round_trip(self, tmp_path):
        metadata = {"NWBFile": {"lab": "Lerner"}, "Ophys": {"FiberPhotometry": {"OpticalFibers": [{"name": "f"}]}}}
        path = tmp_path / "meta.yaml"
        m.dump_yaml(metadata, path)
        assert m.load_yaml(path) == metadata

    def test_loads_blank_text_returns_empty_dict(self):
        assert m.loads_yaml("") == {}

    def test_example_metadata_loads_and_reserializes_stably(self, example_metadata):
        # Dumping then re-loading the bundled template yields an identical dict.
        assert m.loads_yaml(m.dumps_yaml(example_metadata)) == example_metadata
