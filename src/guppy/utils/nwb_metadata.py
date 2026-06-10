"""Pure helpers for the GuPPy NWB-export metadata form.

This module is the single source of truth for the shape of the fiber-photometry
metadata that the neuroconv ``TDTFiberPhotometryGuppyConverter`` consumes. It
knows nothing about neuroconv, Panel, or the GUI -- it only converts between
three representations:

1. the nested metadata dict / YAML the converter expects,
2. flat ``pandas`` DataFrames (one per component list) that drive the GUI's
   editable Tabulator tables, and
3. the scalar NWBFile / Subject fields.

Keeping all flatten/unflatten and split/merge logic here (driven by the
declarative :data:`COMPONENT_SCHEMAS`) means the GUI widgets and the serializer
can never drift, and the whole thing is unit-testable without a browser.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class _RobustDumper(yaml.SafeDumper):
    """A SafeDumper that also handles dict subclasses (neuroconv DeepDict) and numpy scalars/arrays.

    The export step persists the fully-resolved converter metadata for provenance;
    that dict can contain ``DeepDict`` mappings and numpy values that the plain
    ``SafeDumper`` cannot represent.
    """


_RobustDumper.add_multi_representer(dict, yaml.SafeDumper.represent_dict)
_RobustDumper.add_multi_representer(np.generic, lambda dumper, data: dumper.represent_data(data.item()))
_RobustDumper.add_representer(np.ndarray, lambda dumper, data: dumper.represent_data(data.tolist()))


@dataclass(frozen=True)
class ComponentSchema:
    """Describe one fiber-photometry component list for the metadata form.

    Attributes
    ----------
    key : str
        The key under ``Ophys.FiberPhotometry`` this component serializes to.
    category : str
        Human-readable group the component is shown under in the GUI (used to
        bundle related tables into one collapsible card).
    columns : list of str
        Ordered column names for the flat DataFrame. Nested sub-dict fields are
        written with dotted names (e.g. ``"fiber_insertion.depth_in_mm"``) and
        are re-nested on serialize.
    list_columns : set of str
        Columns whose cell values are JSON-encoded lists (e.g.
        ``"wavelength_range_in_nm"`` holding ``"[400, 470]"``).
    container : str
        ``"list"`` (default) serializes records directly to a list under
        ``key``. ``"table_rows"`` wraps the records under ``key`` as
        ``{"name", "description", "rows": records}``.
    """

    key: str
    category: str
    columns: list[str]
    list_columns: set[str] = field(default_factory=set)
    container: str = "list"


COMPONENT_SCHEMAS: dict[str, ComponentSchema] = {
    "OpticalFiberModels": ComponentSchema(
        key="OpticalFiberModels",
        category="Optical Fiber",
        columns=["name", "description", "manufacturer", "model_number", "numerical_aperture", "core_diameter_in_um"],
    ),
    "OpticalFibers": ComponentSchema(
        key="OpticalFibers",
        category="Optical Fiber",
        columns=[
            "name",
            "description",
            "model",
            "serial_number",
            "fiber_insertion.depth_in_mm",
            "fiber_insertion.insertion_position_ap_in_mm",
            "fiber_insertion.insertion_position_ml_in_mm",
            "fiber_insertion.insertion_position_dv_in_mm",
            "fiber_insertion.position_reference",
            "fiber_insertion.hemisphere",
            "fiber_insertion.insertion_angle_pitch_in_deg",
            "fiber_insertion.insertion_angle_yaw_in_deg",
            "fiber_insertion.insertion_angle_roll_in_deg",
        ],
    ),
    "ExcitationSourceModels": ComponentSchema(
        key="ExcitationSourceModels",
        category="Excitation",
        columns=[
            "name",
            "description",
            "manufacturer",
            "model_number",
            "source_type",
            "excitation_mode",
            "wavelength_range_in_nm",
        ],
        list_columns={"wavelength_range_in_nm"},
    ),
    "ExcitationSources": ComponentSchema(
        key="ExcitationSources",
        category="Excitation",
        columns=["name", "description", "model", "power_in_W", "intensity_in_W_per_m2", "exposure_time_in_s"],
    ),
    "PhotodetectorModels": ComponentSchema(
        key="PhotodetectorModels",
        category="Photodetector",
        columns=[
            "name",
            "description",
            "manufacturer",
            "model_number",
            "detector_type",
            "wavelength_range_in_nm",
            "gain",
            "gain_unit",
        ],
        list_columns={"wavelength_range_in_nm"},
    ),
    "Photodetectors": ComponentSchema(
        key="Photodetectors",
        category="Photodetector",
        columns=["name", "description", "model", "serial_number"],
    ),
    "BandOpticalFilterModels": ComponentSchema(
        key="BandOpticalFilterModels",
        category="Optical Filters",
        columns=[
            "name",
            "description",
            "manufacturer",
            "model_number",
            "filter_type",
            "center_wavelength_in_nm",
            "bandwidth_in_nm",
        ],
    ),
    "BandOpticalFilters": ComponentSchema(
        key="BandOpticalFilters",
        category="Optical Filters",
        columns=["name", "description", "model"],
    ),
    "DichroicMirrorModels": ComponentSchema(
        key="DichroicMirrorModels",
        category="Dichroic Mirror",
        columns=[
            "name",
            "description",
            "manufacturer",
            "model_number",
            "cut_on_wavelength_in_nm",
            "reflection_band_in_nm",
            "transmission_band_in_nm",
            "angle_of_incidence_in_degrees",
        ],
        list_columns={"reflection_band_in_nm", "transmission_band_in_nm"},
    ),
    "DichroicMirrors": ComponentSchema(
        key="DichroicMirrors",
        category="Dichroic Mirror",
        columns=["name", "description", "model", "serial_number"],
    ),
    "FiberPhotometryViruses": ComponentSchema(
        key="FiberPhotometryViruses",
        category="Virus & Indicator",
        columns=["name", "description", "manufacturer", "construct_name", "titer_in_vg_per_ml"],
    ),
    "FiberPhotometryVirusInjections": ComponentSchema(
        key="FiberPhotometryVirusInjections",
        category="Virus & Indicator",
        columns=[
            "name",
            "description",
            "viral_vector",
            "location",
            "hemisphere",
            "reference",
            "ap_in_mm",
            "ml_in_mm",
            "dv_in_mm",
            "volume_in_uL",
        ],
    ),
    "FiberPhotometryIndicators": ComponentSchema(
        key="FiberPhotometryIndicators",
        category="Virus & Indicator",
        columns=["name", "description", "manufacturer", "label", "viral_vector_injection"],
    ),
    "CommandedVoltageSeries": ComponentSchema(
        key="CommandedVoltageSeries",
        category="Commanded Voltage",
        columns=["name", "description", "stream_name", "index", "unit", "frequency"],
    ),
    "FiberPhotometryTable": ComponentSchema(
        key="FiberPhotometryTable",
        category="Fiber Photometry Table",
        columns=[
            "name",
            "location",
            "excitation_wavelength_in_nm",
            "emission_wavelength_in_nm",
            "indicator",
            "optical_fiber",
            "excitation_source",
            "commanded_voltage_series",
            "photodetector",
            "dichroic_mirror",
            "emission_filter",
            "excitation_filter",
        ],
        container="table_rows",
    ),
    "FiberPhotometryResponseSeries": ComponentSchema(
        key="FiberPhotometryResponseSeries",
        category="Response Series",
        columns=[
            "name",
            "description",
            "stream_name",
            "stream_indices",
            "unit",
            "fiber_photometry_table_region",
            "fiber_photometry_table_region_description",
        ],
        list_columns={"fiber_photometry_table_region"},
    ),
}

# Scalar (non-table) project-level fields, grouped by the metadata block they
# serialize to. The FiberPhotometryTable name/description are project-level too
# because the table itself (vs its rows) carries the experiment description.
PROJECT_NWBFILE_KEYS = ("lab", "institution", "experimenter")
PROJECT_SUBJECT_KEYS = ("species", "genotype", "strain")
PROJECT_TABLE_KEYS = ("fiber_photometry_table_name", "fiber_photometry_table_description")

# Per-session fields that vary between sessions and live in the per-session YAML.
SESSION_NWBFILE_KEYS = ("session_description", "identifier")
SESSION_SUBJECT_KEYS = ("subject_id", "sex", "age", "date_of_birth")

# experimenter is an NWB list-of-strings; everything else is a plain scalar.
_LIST_SCALAR_KEYS = {"experimenter"}


# ----------------------------------------------------------------------------------------------------------------------
# Record <-> flat row helpers
# ----------------------------------------------------------------------------------------------------------------------
def _is_empty(value: object) -> bool:
    """Return True for values that should be dropped (blank cells, NaN, None)."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _get_nested(record: dict, dotted_key: str) -> object:
    """Read ``record`` at a dotted path, returning None if any level is missing."""
    current: object = record
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _set_nested(target: dict, dotted_key: str, value: object) -> None:
    """Write ``value`` into ``target`` at a dotted path, creating sub-dicts."""
    parts = dotted_key.split(".")
    current = target
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def flatten_record(record: dict, schema: ComponentSchema) -> dict:
    """Flatten a nested metadata record into a flat row keyed by ``schema.columns``.

    Nested sub-dicts become dotted columns and list values become JSON strings,
    so every cell is a primitive the Tabulator can edit directly.
    """
    row: dict = {}
    for column in schema.columns:
        value = _get_nested(record, column)
        if column in schema.list_columns and value is not None:
            value = json.dumps(value)
        row[column] = "" if value is None else value
    return row


def unflatten_record(row: dict, schema: ComponentSchema) -> dict:
    """Rebuild a nested metadata record from a flat row, dropping empty cells.

    Dotted columns are re-nested into sub-dicts; JSON-string list columns are
    parsed back into lists. Empty/NaN cells are omitted so blank optional fields
    never overwrite the converter's auto-filled values during a deep merge.
    """
    record: dict = {}
    for column in schema.columns:
        value = row.get(column)
        if _is_empty(value):
            continue
        if column in schema.list_columns:
            try:
                value = json.loads(value) if isinstance(value, str) else value
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Column {column!r} must be a JSON list (e.g. '[400, 470]'); got {value!r}: {exc}"
                ) from exc
        _set_nested(record, column, value)
    return record


def records_to_dataframe(records: list[dict], schema: ComponentSchema) -> pd.DataFrame:
    """Convert a list of nested metadata records into a flat DataFrame for editing."""
    rows = [flatten_record(record, schema) for record in records]
    return pd.DataFrame(rows, columns=list(schema.columns))


def dataframe_to_records(dataframe: pd.DataFrame, schema: ComponentSchema) -> list[dict]:
    """Convert an edited Tabulator DataFrame back into nested metadata records.

    Fully-empty rows are dropped (they carry no information after empty-cell
    pruning), which lets the GUI keep a trailing blank row for "add new".
    """
    records = []
    for _, row in dataframe.iterrows():
        record = unflatten_record(row.to_dict(), schema)
        if record:
            records.append(record)
    return records


# ----------------------------------------------------------------------------------------------------------------------
# Project / session metadata assembly
# ----------------------------------------------------------------------------------------------------------------------
def _drop_empty(mapping: dict) -> dict:
    """Return ``mapping`` without keys whose values are empty."""
    return {key: value for key, value in mapping.items() if not _is_empty(value)}


def build_project_metadata_dict(component_dataframes: dict[str, pd.DataFrame], scalars: dict) -> dict:
    """Assemble the project-level metadata overlay from edited tables and scalars.

    Parameters
    ----------
    component_dataframes : dict
        Maps each ``COMPONENT_SCHEMAS`` key to its edited DataFrame.
    scalars : dict
        Project scalar values (``PROJECT_NWBFILE_KEYS``, ``PROJECT_SUBJECT_KEYS``,
        ``PROJECT_TABLE_KEYS``).

    Returns
    -------
    dict
        Nested metadata with ``NWBFile``, ``Subject`` and ``Ophys.FiberPhotometry``
        blocks. Empty scalars and empty component lists are omitted.
    """
    fiber_photometry: dict = {}
    for key, schema in COMPONENT_SCHEMAS.items():
        dataframe = component_dataframes.get(key)
        if dataframe is None:
            continue
        records = dataframe_to_records(dataframe, schema)
        if not records:
            continue
        if schema.container == "table_rows":
            table: dict = _drop_empty(
                {
                    "name": scalars.get("fiber_photometry_table_name"),
                    "description": scalars.get("fiber_photometry_table_description"),
                }
            )
            table["rows"] = records
            fiber_photometry[key] = table
        else:
            fiber_photometry[key] = records

    nwbfile = _drop_empty({key: scalars.get(key) for key in PROJECT_NWBFILE_KEYS})
    subject = _drop_empty({key: scalars.get(key) for key in PROJECT_SUBJECT_KEYS})

    metadata: dict = {}
    if nwbfile:
        metadata["NWBFile"] = nwbfile
    if subject:
        metadata["Subject"] = subject
    if fiber_photometry:
        metadata["Ophys"] = {"FiberPhotometry": fiber_photometry}
    return metadata


def parse_project_metadata_dict(metadata: dict) -> tuple[dict[str, pd.DataFrame], dict]:
    """Inverse of :func:`build_project_metadata_dict` for populating the GUI from a dict."""
    fiber_photometry = metadata.get("Ophys", {}).get("FiberPhotometry", {})
    component_dataframes: dict[str, pd.DataFrame] = {}
    for key, schema in COMPONENT_SCHEMAS.items():
        block = fiber_photometry.get(key)
        if block is None:
            records: list[dict] = []
        elif schema.container == "table_rows":
            records = block.get("rows", [])
        else:
            records = block
        component_dataframes[key] = records_to_dataframe(records, schema)

    nwbfile = metadata.get("NWBFile", {})
    subject = metadata.get("Subject", {})
    table = fiber_photometry.get("FiberPhotometryTable", {})
    scalars: dict = {}
    for key in PROJECT_NWBFILE_KEYS:
        scalars[key] = nwbfile.get(key, "")
    for key in PROJECT_SUBJECT_KEYS:
        scalars[key] = subject.get(key, "")
    scalars["fiber_photometry_table_name"] = table.get("name", "")
    scalars["fiber_photometry_table_description"] = table.get("description", "")
    return component_dataframes, scalars


def build_session_metadata_dict(scalars: dict) -> dict:
    """Assemble the per-session metadata overlay from scalar fields."""
    nwbfile = _drop_empty({key: scalars.get(key) for key in SESSION_NWBFILE_KEYS})
    subject = _drop_empty({key: scalars.get(key) for key in SESSION_SUBJECT_KEYS})
    metadata: dict = {}
    if nwbfile:
        metadata["NWBFile"] = nwbfile
    if subject:
        metadata["Subject"] = subject
    return metadata


def parse_session_metadata_dict(metadata: dict) -> dict:
    """Inverse of :func:`build_session_metadata_dict`."""
    nwbfile = metadata.get("NWBFile", {})
    subject = metadata.get("Subject", {})
    scalars: dict = {}
    for key in SESSION_NWBFILE_KEYS:
        scalars[key] = nwbfile.get(key, "")
    for key in SESSION_SUBJECT_KEYS:
        scalars[key] = subject.get(key, "")
    return scalars


def split_template_into_project_and_session(full_metadata: dict) -> tuple[dict, dict]:
    """Partition a full metadata dict into project-level and per-session overlays.

    The project overlay keeps the entire ``Ophys`` block plus the project NWBFile
    and Subject keys; the session overlay keeps only the per-session NWBFile and
    Subject keys.
    """
    nwbfile = full_metadata.get("NWBFile", {})
    subject = full_metadata.get("Subject", {})

    project: dict = {}
    if "Ophys" in full_metadata:
        project["Ophys"] = full_metadata["Ophys"]
    project_nwbfile = {key: nwbfile[key] for key in PROJECT_NWBFILE_KEYS if key in nwbfile}
    project_subject = {key: subject[key] for key in PROJECT_SUBJECT_KEYS if key in subject}
    if project_nwbfile:
        project["NWBFile"] = project_nwbfile
    if project_subject:
        project["Subject"] = project_subject

    session: dict = {}
    session_nwbfile = {key: nwbfile[key] for key in SESSION_NWBFILE_KEYS if key in nwbfile}
    session_subject = {key: subject[key] for key in SESSION_SUBJECT_KEYS if key in subject}
    if session_nwbfile:
        session["NWBFile"] = session_nwbfile
    if session_subject:
        session["Subject"] = session_subject

    return project, session


def merge_metadata(*metadata_dicts: dict) -> dict:
    """Deep-merge metadata dicts left-to-right (later wins).

    Nested dicts are merged recursively; every other value (including lists) is
    replaced wholesale. Used for previewing/testing the project<-session merge;
    the export step itself uses neuroconv's ``dict_deep_update`` against the
    converter's auto-filled metadata.
    """
    merged: dict = {}
    for metadata in metadata_dicts:
        _deep_update(merged, metadata)
    return merged


def _deep_update(target: dict, source: dict) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


# ----------------------------------------------------------------------------------------------------------------------
# YAML I/O
# ----------------------------------------------------------------------------------------------------------------------
def load_yaml(path: str | Path) -> dict:
    """Load a YAML file into a dict."""
    with open(path) as yaml_file:
        return yaml.safe_load(yaml_file) or {}


def dump_yaml(metadata: dict, path: str | Path) -> None:
    """Write ``metadata`` to ``path`` as human-editable YAML (insertion order preserved)."""
    with open(path, "w") as yaml_file:
        yaml.dump(metadata, yaml_file, Dumper=_RobustDumper, sort_keys=False, default_flow_style=False)


def loads_yaml(text: str) -> dict:
    """Parse a YAML string into a dict (empty/blank text yields an empty dict)."""
    return yaml.safe_load(text) or {}


def dumps_yaml(metadata: dict) -> str:
    """Serialize ``metadata`` to a human-editable YAML string (insertion order preserved)."""
    return yaml.dump(metadata, Dumper=_RobustDumper, sort_keys=False, default_flow_style=False)
