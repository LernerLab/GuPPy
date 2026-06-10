"""Unit tests for the pure NWB-export metadata helpers."""

import importlib.resources

import pandas as pd
import pytest

from guppy.utils import nwb_metadata as m


@pytest.fixture
def bundled_template() -> dict:
    text = importlib.resources.files("guppy").joinpath("resources/fiber_photometry_metadata_template.yaml").read_text()
    return m.loads_yaml(text)


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


class TestProjectMetadata:
    def test_build_project_metadata_dict_assembles_blocks(self):
        component_dataframes = {key: m.records_to_dataframe([], schema) for key, schema in m.COMPONENT_SCHEMAS.items()}
        component_dataframes["OpticalFibers"] = m.records_to_dataframe(
            [{"name": "optical_fiber", "fiber_insertion": {"depth_in_mm": 2.8}}],
            m.COMPONENT_SCHEMAS["OpticalFibers"],
        )
        scalars = {
            "lab": "Lerner Lab",
            "institution": "",
            "experimenter": ["Doe, Jane"],
            "species": "Mus musculus",
            "genotype": "",
            "strain": "",
            "fiber_photometry_table_name": "fiber_photometry_table",
            "fiber_photometry_table_description": "desc",
        }
        metadata = m.build_project_metadata_dict(component_dataframes, scalars)
        assert metadata["NWBFile"] == {"lab": "Lerner Lab", "experimenter": ["Doe, Jane"]}
        assert metadata["Subject"] == {"species": "Mus musculus"}
        assert metadata["Ophys"]["FiberPhotometry"]["OpticalFibers"] == [
            {"name": "optical_fiber", "fiber_insertion": {"depth_in_mm": 2.8}}
        ]
        # Empty component lists are omitted entirely.
        assert "Photodetectors" not in metadata["Ophys"]["FiberPhotometry"]

    def test_project_round_trip_preserves_template_components(self, bundled_template):
        project, _ = m.split_template_into_project_and_session(bundled_template)
        dataframes, scalars = m.parse_project_metadata_dict(project)
        rebuilt = m.build_project_metadata_dict(dataframes, scalars)
        original_fp = bundled_template["Ophys"]["FiberPhotometry"]
        rebuilt_fp = rebuilt["Ophys"]["FiberPhotometry"]
        # Nested records, list fields, and the table all survive the round-trip.
        assert rebuilt_fp["OpticalFibers"] == original_fp["OpticalFibers"]
        assert rebuilt_fp["DichroicMirrorModels"] == original_fp["DichroicMirrorModels"]
        assert rebuilt_fp["FiberPhotometryTable"] == original_fp["FiberPhotometryTable"]

    def test_parse_project_metadata_dict_recovers_table_scalars(self, bundled_template):
        project, _ = m.split_template_into_project_and_session(bundled_template)
        _, scalars = m.parse_project_metadata_dict(project)
        assert scalars["fiber_photometry_table_name"] == "fiber_photometry_table"


class TestSessionMetadata:
    def test_build_and_parse_session_round_trip(self):
        scalars = {
            "session_description": "RI30 session",
            "identifier": "Photo_63_207",
            "subject_id": "63_207",
            "sex": "M",
            "age": "P90D",
            "date_of_birth": "",
        }
        metadata = m.build_session_metadata_dict(scalars)
        assert metadata == {
            "NWBFile": {"session_description": "RI30 session", "identifier": "Photo_63_207"},
            "Subject": {"subject_id": "63_207", "sex": "M", "age": "P90D"},
        }
        recovered = m.parse_session_metadata_dict(metadata)
        assert recovered["subject_id"] == "63_207"
        assert recovered["date_of_birth"] == ""


class TestSplit:
    def test_split_partitions_project_and_session_keys(self):
        full = {
            "NWBFile": {"lab": "Lerner", "session_description": "s", "identifier": "id"},
            "Subject": {"species": "Mus musculus", "subject_id": "63", "sex": "M"},
            "Ophys": {"FiberPhotometry": {"OpticalFibers": [{"name": "f"}]}},
        }
        project, session = m.split_template_into_project_and_session(full)
        assert project["NWBFile"] == {"lab": "Lerner"}
        assert project["Subject"] == {"species": "Mus musculus"}
        assert project["Ophys"] == {"FiberPhotometry": {"OpticalFibers": [{"name": "f"}]}}
        assert session["NWBFile"] == {"session_description": "s", "identifier": "id"}
        assert session["Subject"] == {"subject_id": "63", "sex": "M"}


class TestMerge:
    def test_merge_precedence_session_over_project_over_base(self):
        base = {"NWBFile": {"lab": "Base", "institution": "Inst"}, "Subject": {"species": "Mus"}}
        project = {"NWBFile": {"lab": "Project"}}
        session = {"NWBFile": {"identifier": "id"}, "Subject": {"subject_id": "63"}}
        merged = m.merge_metadata(base, project, session)
        assert merged["NWBFile"] == {"lab": "Project", "institution": "Inst", "identifier": "id"}
        assert merged["Subject"] == {"species": "Mus", "subject_id": "63"}

    def test_merge_replaces_lists_wholesale(self):
        merged = m.merge_metadata({"a": [1, 2, 3]}, {"a": [9]})
        assert merged["a"] == [9]


class TestYamlIO:
    def test_dump_then_load_round_trip(self, tmp_path):
        metadata = {"NWBFile": {"lab": "Lerner"}, "Ophys": {"FiberPhotometry": {"OpticalFibers": [{"name": "f"}]}}}
        path = tmp_path / "meta.yaml"
        m.dump_yaml(metadata, path)
        assert m.load_yaml(path) == metadata

    def test_loads_blank_text_returns_empty_dict(self):
        assert m.loads_yaml("") == {}

    def test_bundled_template_loads_and_reserializes_stably(self, bundled_template):
        # Dumping then re-loading the bundled template yields an identical dict.
        assert m.loads_yaml(m.dumps_yaml(bundled_template)) == bundled_template
