"""Pure core for the GuPPy NWB-export metadata form (channel-centric, schema-driven).

GuPPy owns the fiber-photometry channel structure (it wrote ``storesList.csv``),
so the form auto-derives everything structural and the user only supplies what
GuPPy cannot know: the hardware/biology device library, per-channel wavelengths,
optional coordinates, and which device each channel used.

This module knows nothing about Panel or neuroconv. It provides:

- :func:`derive_channels` -- the ordered fiber-photometry channels from a session's
  ``storesList.csv`` (reusing GuPPy's own :func:`get_control_and_signal_channel_names`).
- :data:`CATEGORIES` + :func:`field_specs` -- a schema-driven description of each
  device category (fields, required flags, docs, links), introspected from the
  *installed* ndx-ophys-devices / ndx-fiber-photometry extensions (which is what the
  converter validates against).
- :func:`build_metadata_dict` / :func:`parse_metadata_dict` -- convert between the
  form state and the converter's metadata dict, generating the FiberPhotometryTable
  rows and FiberPhotometryResponseSeries from the channels.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from ..analysis.io_utils import get_control_and_signal_channel_names


class _RobustDumper(yaml.SafeDumper):
    """SafeDumper that also handles dict subclasses (DeepDict) and numpy scalars/arrays."""


_RobustDumper.add_multi_representer(dict, yaml.SafeDumper.represent_dict)
_RobustDumper.add_multi_representer(np.generic, lambda dumper, data: dumper.represent_data(data.item()))
_RobustDumper.add_representer(np.ndarray, lambda dumper, data: dumper.represent_data(data.tolist()))


# ----------------------------------------------------------------------------------------------------------------------
# Channels (GuPPy-native, from storesList.csv)
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Channel:
    """One fiber-photometry channel derived from ``storesList.csv``."""

    region: str
    role: str  # "signal" or "control"
    store_name: str

    @property
    def response_series_name(self) -> str:
        """Conventional FiberPhotometryResponseSeries name for this channel."""
        suffix = "calcium_signal" if self.role == "signal" else "isosbestic_control"
        return f"{self.region}_{suffix}"


def derive_channels(output_dir: str | Path) -> list[Channel]:
    """Return the ordered fiber-photometry channels for a GuPPy output directory.

    Reads ``<output_dir>/storesList.csv`` and pairs signal/control names per region
    using GuPPy's own :func:`get_control_and_signal_channel_names`. Order is canonical
    (region-sorted, control then signal within a region) so it matches between the
    form (table rows) and the generated response series.
    """
    stores_list = np.genfromtxt(os.path.join(output_dir, "storesList.csv"), dtype="str", delimiter=",").reshape(2, -1)
    store_names = stores_list[0, :]
    semantic_names = stores_list[1, :]
    semantic_to_store = {semantic: store for store, semantic in zip(store_names, semantic_names)}

    paired = get_control_and_signal_channel_names(stores_list)  # shape (2, N): row 0 control, row 1 signal
    channels: list[Channel] = []
    for column in range(paired.shape[1]):
        control_name = paired[0, column]
        signal_name = paired[1, column]
        region = str(signal_name[len("signal_") :]).lower()
        channels.append(Channel(region=region, role="control", store_name=str(semantic_to_store[control_name])))
        channels.append(Channel(region=region, role="signal", store_name=str(semantic_to_store[signal_name])))
    return channels


# ----------------------------------------------------------------------------------------------------------------------
# Device-library schema (introspected from the installed extensions)
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldSpec:
    """One editable field of a device entry, derived from the ndx spec."""

    name: str
    required: bool
    doc: str
    dtype: str  # "text" | "float"
    is_list: bool  # spec shape [2] -> a pair of numbers
    target: str  # "model" | "instance" | "insertion" | "self"
    link_target: str | None = None  # category key whose names populate a dropdown


@dataclass(frozen=True)
class CategorySpec:
    """A user-facing device category and how it maps to ndx metadata keys."""

    key: str
    label: str
    kind: str  # "merged" (model + instance) or "single"
    list_key: str  # metadata key for the (single) or instance list
    model_type: str | None = None
    instance_type: str | None = None
    model_list_key: str | None = None
    single_type: str | None = None
    has_insertion: bool = False
    links: dict[str, str] = field(default_factory=dict)  # link field name -> category key


CATEGORIES: dict[str, CategorySpec] = {
    "optical_fiber": CategorySpec(
        key="optical_fiber",
        label="Optical fibers",
        kind="merged",
        model_type="OpticalFiberModel",
        instance_type="OpticalFiber",
        model_list_key="OpticalFiberModels",
        list_key="OpticalFibers",
        has_insertion=True,
    ),
    "excitation_source": CategorySpec(
        key="excitation_source",
        label="Excitation sources",
        kind="merged",
        model_type="ExcitationSourceModel",
        instance_type="ExcitationSource",
        model_list_key="ExcitationSourceModels",
        list_key="ExcitationSources",
    ),
    "photodetector": CategorySpec(
        key="photodetector",
        label="Photodetectors",
        kind="merged",
        model_type="PhotodetectorModel",
        instance_type="Photodetector",
        model_list_key="PhotodetectorModels",
        list_key="Photodetectors",
    ),
    "band_optical_filter": CategorySpec(
        key="band_optical_filter",
        label="Band optical filters",
        kind="merged",
        model_type="BandOpticalFilterModel",
        instance_type="BandOpticalFilter",
        model_list_key="BandOpticalFilterModels",
        list_key="BandOpticalFilters",
    ),
    "dichroic_mirror": CategorySpec(
        key="dichroic_mirror",
        label="Dichroic mirrors",
        kind="merged",
        model_type="DichroicMirrorModel",
        instance_type="DichroicMirror",
        model_list_key="DichroicMirrorModels",
        list_key="DichroicMirrors",
    ),
    "indicator": CategorySpec(
        key="indicator",
        label="Indicators",
        kind="single",
        single_type="Indicator",
        list_key="FiberPhotometryIndicators",
        links={"viral_vector_injection": "virus_injection"},
    ),
    "virus": CategorySpec(
        key="virus",
        label="Viruses",
        kind="single",
        single_type="ViralVector",
        list_key="FiberPhotometryViruses",
    ),
    "virus_injection": CategorySpec(
        key="virus_injection",
        label="Virus injections",
        kind="single",
        single_type="ViralVectorInjection",
        list_key="FiberPhotometryVirusInjections",
        links={"viral_vector": "virus"},
    ),
}

# Channel-table link columns -> device category supplying the dropdown options.
CHANNEL_LINKS: dict[str, str] = {
    "indicator": "indicator",
    "optical_fiber": "optical_fiber",
    "excitation_source": "excitation_source",
    "photodetector": "photodetector",
    "dichroic_mirror": "dichroic_mirror",
    "excitation_filter": "band_optical_filter",
    "emission_filter": "band_optical_filter",
}
CHANNEL_REQUIRED_LINKS = ("indicator", "optical_fiber", "excitation_source", "photodetector")
CHANNEL_WAVELENGTHS = ("excitation_wavelength_in_nm", "emission_wavelength_in_nm")

# Instance attributes that are deprecated (they belong on the model) or duplicated.
_INSTANCE_SKIP = {"manufacturer", "model_number", "model_name", "description"}


@lru_cache(maxsize=None)
def _type_attrs(type_name: str) -> tuple[tuple[str, bool, str, str, bool], ...]:
    """Introspect an installed ndx type: ordered (name, required, doc, dtype, is_list)."""
    import ndx_fiber_photometry  # noqa: F401  (register extensions)
    import ndx_ophys_devices  # noqa: F401
    from pynwb import get_type_map

    catalog = get_type_map().namespace_catalog
    spec = None
    for namespace in ("ndx-ophys-devices", "ndx-fiber-photometry"):
        try:
            spec = catalog.get_spec(namespace, type_name)
            break
        except ValueError:
            continue
    assert spec is not None, f"Type {type_name!r} not found in installed ndx namespaces."

    rows = []
    for attribute in spec.attributes:
        dtype = "float" if attribute.dtype == "float" else "text"
        is_list = getattr(attribute, "shape", None) is not None
        rows.append((attribute.name, bool(attribute.required), attribute.doc, dtype, is_list))
    return tuple(rows)


@lru_cache(maxsize=None)
def _type_links(type_name: str) -> dict[str, bool]:
    """Return {link_name: required} for an installed ndx type."""
    import ndx_fiber_photometry  # noqa: F401
    import ndx_ophys_devices  # noqa: F401
    from pynwb import get_type_map

    catalog = get_type_map().namespace_catalog
    for namespace in ("ndx-ophys-devices", "ndx-fiber-photometry"):
        try:
            spec = catalog.get_spec(namespace, type_name)
            break
        except ValueError:
            continue
    return {
        link.name: str(getattr(link, "quantity", "?")) in ("1", "+") for link in (getattr(spec, "links", None) or [])
    }


def field_specs(category_key: str) -> list[FieldSpec]:
    """Return the ordered editable fields for a device category."""
    category = CATEGORIES[category_key]
    specs: list[FieldSpec] = [
        FieldSpec(
            "name", True, "Unique name for this device (referenced by the channel links).", "text", False, "self"
        ),
        FieldSpec("description", False, "Free-form description of the device.", "text", False, "instance"),
    ]

    if category.kind == "merged":
        for name, required, doc, dtype, is_list in _type_attrs(category.model_type):
            if name == "description":
                continue
            specs.append(FieldSpec(name, required, doc, dtype, is_list, "model"))
        for name, required, doc, dtype, is_list in _type_attrs(category.instance_type):
            if name in _INSTANCE_SKIP:
                continue
            specs.append(FieldSpec(name, required, doc, dtype, is_list, "instance"))
        if category.has_insertion:
            for name, required, doc, dtype, is_list in _type_attrs("FiberInsertion"):
                specs.append(FieldSpec(name, required, doc, dtype, is_list, "insertion"))
    else:  # single object (Indicator / ViralVector / ViralVectorInjection)
        specs[1] = FieldSpec("description", False, "Free-form description.", "text", False, "self")
        for name, required, doc, dtype, is_list in _type_attrs(category.single_type):
            if name == "description":
                continue
            specs.append(FieldSpec(name, required, doc, dtype, is_list, "self"))
        link_required = _type_links(category.single_type)
        for link_name, target_category in category.links.items():
            specs.append(
                FieldSpec(
                    link_name,
                    link_required.get(link_name, False),
                    f"Link to a defined {CATEGORIES[target_category].label.rstrip('s').lower()}.",
                    "text",
                    False,
                    "self",
                    link_target=target_category,
                )
            )
    return specs


# ----------------------------------------------------------------------------------------------------------------------
# Scalar (NWBFile / Subject) fields
# ----------------------------------------------------------------------------------------------------------------------
NWBFILE_KEYS = ("session_description", "identifier", "lab", "institution", "experimenter")
SUBJECT_KEYS = ("subject_id", "sex", "age", "date_of_birth", "species", "genotype", "strain")


def _is_empty(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, tuple, dict)) and len(value) == 0:
        return True
    return False


def _drop_empty(mapping: dict) -> dict:
    return {key: value for key, value in mapping.items() if not _is_empty(value)}


# ----------------------------------------------------------------------------------------------------------------------
# Device serialization (merged form entry <-> Model + instance dicts)
# ----------------------------------------------------------------------------------------------------------------------
def _serialize_device(category: CategorySpec, entry: dict) -> tuple[dict | None, dict]:
    """Serialize one form entry into (model_dict | None, object_dict)."""
    specs = field_specs(category.key)
    name = entry.get("name", "")

    if category.kind == "single":
        single = _drop_empty({spec.name: entry.get(spec.name) for spec in specs})
        return None, single

    model_name = f"{name}_model"
    model = {"name": model_name}
    instance = {"name": name, "model": model_name}
    insertion: dict = {}
    for spec in specs:
        if spec.name in ("name",):
            continue
        value = entry.get(spec.name)
        if _is_empty(value):
            continue
        if spec.target == "model":
            model[spec.name] = value
        elif spec.target == "insertion":
            insertion[spec.name] = value
        else:  # instance (description, serial_number)
            instance[spec.name] = value
    if insertion:
        instance["fiber_insertion"] = insertion
    return model, instance


def _deserialize_device(category: CategorySpec, object_dict: dict, models_by_name: dict[str, dict]) -> dict:
    """Recombine a Model + instance (or single object) back into one form entry."""
    if category.kind == "single":
        return dict(object_dict)

    entry: dict = {"name": object_dict.get("name", "")}
    if "description" in object_dict:
        entry["description"] = object_dict["description"]
    if "serial_number" in object_dict:
        entry["serial_number"] = object_dict["serial_number"]
    for sub_key, sub_value in (object_dict.get("fiber_insertion") or {}).items():
        entry[sub_key] = sub_value
    model = models_by_name.get(object_dict.get("model", ""), {})
    for model_key, model_value in model.items():
        if model_key != "name":
            entry[model_key] = model_value
    return entry


# ----------------------------------------------------------------------------------------------------------------------
# build / parse
# ----------------------------------------------------------------------------------------------------------------------
def build_metadata_dict(
    devices: dict[str, list[dict]], channel_rows: list[dict], scalars: dict, channels: list[Channel]
) -> dict:
    """Assemble the full session metadata overlay from the form state.

    Generates the FiberPhotometryTable rows (one per channel, ``location`` = region)
    and the FiberPhotometryResponseSeries (one per channel) from ``channels``.
    """
    fiber_photometry: dict = {}
    for category in CATEGORIES.values():
        models: list[dict] = []
        objects: list[dict] = []
        for entry in devices.get(category.key, []):
            if _is_empty(entry.get("name")):
                continue
            model, object_dict = _serialize_device(category, entry)
            if model is not None:
                models.append(model)
            objects.append(object_dict)
        if not objects:
            continue
        if category.model_list_key:
            fiber_photometry[category.model_list_key] = models
        fiber_photometry[category.list_key] = objects

    table_rows = []
    response_series = []
    for index, channel in enumerate(channels):
        row = {"name": index, "location": channel.region}
        annotations = channel_rows[index] if index < len(channel_rows) else {}
        row.update(_drop_empty({key: annotations.get(key) for key in (*CHANNEL_WAVELENGTHS, *CHANNEL_LINKS)}))
        table_rows.append(row)
        response_series.append(
            {
                "name": channel.response_series_name,
                "description": f"Fluorescence from the {channel.region} {channel.role} channel.",
                "stream_name": channel.store_name,
                "stream_indices": None,
                "unit": "a.u.",
                "fiber_photometry_table_region": [index],
                "fiber_photometry_table_region_description": (
                    f"FiberPhotometryTable row for the {channel.region} {channel.role} channel."
                ),
            }
        )
    if table_rows:
        fiber_photometry["FiberPhotometryTable"] = {
            "name": scalars.get("fiber_photometry_table_name") or "fiber_photometry_table",
            "description": scalars.get("fiber_photometry_table_description") or "Fiber photometry table.",
            "rows": table_rows,
        }
        fiber_photometry["FiberPhotometryResponseSeries"] = response_series

    nwbfile = _drop_empty({key: scalars.get(key) for key in NWBFILE_KEYS})
    subject = _drop_empty({key: scalars.get(key) for key in SUBJECT_KEYS})

    metadata: dict = {}
    if nwbfile:
        metadata["NWBFile"] = nwbfile
    if subject:
        metadata["Subject"] = subject
    if fiber_photometry:
        metadata["Ophys"] = {"FiberPhotometry": fiber_photometry}
    return metadata


def parse_metadata_dict(metadata: dict, channels: list[Channel]) -> tuple[dict[str, list[dict]], list[dict], dict]:
    """Inverse of :func:`build_metadata_dict` for repopulating the form from a saved dict."""
    fiber_photometry = metadata.get("Ophys", {}).get("FiberPhotometry", {})

    devices: dict[str, list[dict]] = {}
    for category in CATEGORIES.values():
        models_by_name = {model["name"]: model for model in fiber_photometry.get(category.model_list_key or "", [])}
        objects = fiber_photometry.get(category.list_key, [])
        devices[category.key] = [_deserialize_device(category, obj, models_by_name) for obj in objects]

    saved_rows = fiber_photometry.get("FiberPhotometryTable", {}).get("rows", [])
    channel_rows: list[dict] = []
    for index, _channel in enumerate(channels):
        row = saved_rows[index] if index < len(saved_rows) else {}
        channel_rows.append({key: row.get(key, "") for key in (*CHANNEL_WAVELENGTHS, *CHANNEL_LINKS)})

    nwbfile = metadata.get("NWBFile", {})
    subject = metadata.get("Subject", {})
    table = fiber_photometry.get("FiberPhotometryTable", {})
    scalars: dict = {key: nwbfile.get(key, "") for key in NWBFILE_KEYS}
    scalars.update({key: subject.get(key, "") for key in SUBJECT_KEYS})
    scalars["fiber_photometry_table_name"] = table.get("name", "")
    scalars["fiber_photometry_table_description"] = table.get("description", "")
    return devices, channel_rows, scalars


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
    """Parse a YAML string into a dict (blank text yields an empty dict)."""
    return yaml.safe_load(text) or {}


def dumps_yaml(metadata: dict) -> str:
    """Serialize ``metadata`` to a human-editable YAML string (insertion order preserved)."""
    return yaml.dump(metadata, Dumper=_RobustDumper, sort_keys=False, default_flow_style=False)
