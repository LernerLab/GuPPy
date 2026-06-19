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
    target: str  # "self" | "insertion"
    link_target: str | None = None  # category key whose names populate a dropdown
    options: tuple[str, ...] | None = None  # enumerated choices -> rendered as a dropdown


@dataclass(frozen=True)
class CategorySpec:
    """A user-facing device category and how it maps to one ndx metadata list."""

    key: str
    label: str  # plural section title
    singular: str  # used for the "Add <singular>" button
    ndx_type: str  # installed type to introspect for fields/docs/required
    list_key: str  # metadata key under Ophys.FiberPhotometry
    has_insertion: bool = False  # OpticalFiber: emit a (mandatory) fiber_insertion group
    skip_deprecated: bool = False  # device instances: drop the deprecated model-ish attrs
    links: dict[str, str] = field(default_factory=dict)  # link field name -> target category key


# Ordered by dependency: a model before the instance that links to it; viral vector before its
# injection before the indicator that references the injection. Instances are what channels link to.
CATEGORIES: dict[str, CategorySpec] = {
    "optical_fiber_model": CategorySpec(
        "optical_fiber_model", "Optical fiber models", "optical fiber model", "OpticalFiberModel", "OpticalFiberModels"
    ),
    "optical_fiber": CategorySpec(
        "optical_fiber",
        "Optical fibers",
        "optical fiber",
        "OpticalFiber",
        "OpticalFibers",
        has_insertion=True,
        skip_deprecated=True,
        links={"model": "optical_fiber_model"},
    ),
    "excitation_source_model": CategorySpec(
        "excitation_source_model",
        "Excitation source models",
        "excitation source model",
        "ExcitationSourceModel",
        "ExcitationSourceModels",
    ),
    "excitation_source": CategorySpec(
        "excitation_source",
        "Excitation sources",
        "excitation source",
        "ExcitationSource",
        "ExcitationSources",
        skip_deprecated=True,
        links={"model": "excitation_source_model"},
    ),
    "photodetector_model": CategorySpec(
        "photodetector_model",
        "Photodetector models",
        "photodetector model",
        "PhotodetectorModel",
        "PhotodetectorModels",
    ),
    "photodetector": CategorySpec(
        "photodetector",
        "Photodetectors",
        "photodetector",
        "Photodetector",
        "Photodetectors",
        skip_deprecated=True,
        links={"model": "photodetector_model"},
    ),
    "band_optical_filter_model": CategorySpec(
        "band_optical_filter_model",
        "Band optical filter models",
        "band optical filter model",
        "BandOpticalFilterModel",
        "BandOpticalFilterModels",
    ),
    "band_optical_filter": CategorySpec(
        "band_optical_filter",
        "Band optical filters",
        "band optical filter",
        "BandOpticalFilter",
        "BandOpticalFilters",
        skip_deprecated=True,
        links={"model": "band_optical_filter_model"},
    ),
    "dichroic_mirror_model": CategorySpec(
        "dichroic_mirror_model",
        "Dichroic mirror models",
        "dichroic mirror model",
        "DichroicMirrorModel",
        "DichroicMirrorModels",
    ),
    "dichroic_mirror": CategorySpec(
        "dichroic_mirror",
        "Dichroic mirrors",
        "dichroic mirror",
        "DichroicMirror",
        "DichroicMirrors",
        skip_deprecated=True,
        links={"model": "dichroic_mirror_model"},
    ),
    "virus": CategorySpec("virus", "Viruses", "virus", "ViralVector", "FiberPhotometryViruses"),
    "virus_injection": CategorySpec(
        "virus_injection",
        "Virus injections",
        "virus injection",
        "ViralVectorInjection",
        "FiberPhotometryVirusInjections",
        links={"viral_vector": "virus"},
    ),
    "indicator": CategorySpec(
        "indicator",
        "Indicators",
        "indicator",
        "Indicator",
        "FiberPhotometryIndicators",
        links={"viral_vector_injection": "virus_injection"},
    ),
}

# Channel-table link columns -> the (instance) device category supplying the dropdown options.
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

# Deprecated instance attributes that belong on the model (dropped from instance forms).
_DEPRECATED_INSTANCE_ATTRS = {"manufacturer", "model_number", "model_name"}

# Free-text-in-the-spec fields whose values are effectively enumerated (rendered as dropdowns).
ENUM_OPTIONS: dict[str, tuple[str, ...]] = {
    "source_type": ("LED", "Gas Laser", "Solid-State Laser"),
    "excitation_mode": ("one-photon", "two-photon", "three-photon", "other"),
    "detector_type": ("photodiode", "PMT", "CCD", "CMOS", "EBCCD", "intensified CCD", "FTIR"),
    "filter_type": ("Bandpass", "Bandstop"),
    "hemisphere": ("left", "right"),
}


@lru_cache(maxsize=None)
def _type_spec(type_name: str):  # noqa: ANN202  (returns an hdmf Spec)
    import ndx_fiber_photometry  # noqa: F401  (register extensions)
    import ndx_ophys_devices  # noqa: F401
    from pynwb import get_type_map

    catalog = get_type_map().namespace_catalog
    for namespace in ("ndx-ophys-devices", "ndx-fiber-photometry"):
        try:
            return catalog.get_spec(namespace, type_name)
        except ValueError:
            continue
    raise AssertionError(f"Type {type_name!r} not found in installed ndx namespaces.")


def _type_attrs(type_name: str) -> tuple[tuple[str, bool, str, str, bool], ...]:
    """Introspect an installed ndx type: ordered (name, required, doc, dtype, is_list)."""
    rows = []
    for attribute in _type_spec(type_name).attributes:
        dtype = "float" if attribute.dtype == "float" else "text"
        is_list = getattr(attribute, "shape", None) is not None
        rows.append((attribute.name, bool(attribute.required), attribute.doc, dtype, is_list))
    return tuple(rows)


def _type_links(type_name: str) -> dict[str, bool]:
    """Return {link_name: required} for an installed ndx type."""
    spec = _type_spec(type_name)
    return {
        link.name: str(getattr(link, "quantity", "?")) in ("1", "+") for link in (getattr(spec, "links", None) or [])
    }


def field_specs(category_key: str) -> list[FieldSpec]:
    """Return the ordered editable fields for a device category."""
    category = CATEGORIES[category_key]
    specs: list[FieldSpec] = [
        FieldSpec("name", True, "Unique name for this object (referenced by links).", "text", False, "self"),
        FieldSpec("description", False, "Free-form description.", "text", False, "self"),
    ]
    for name, required, doc, dtype, is_list in _type_attrs(category.ndx_type):
        if name == "description":
            continue
        if category.skip_deprecated and name in _DEPRECATED_INSTANCE_ATTRS:
            continue
        specs.append(FieldSpec(name, required, doc, dtype, is_list, "self", options=ENUM_OPTIONS.get(name)))
    if category.has_insertion:
        # The fiber_insertion group itself is mandatory (quantity 1); its individual coordinates are optional.
        for name, required, doc, dtype, is_list in _type_attrs("FiberInsertion"):
            specs.append(FieldSpec(name, required, doc, dtype, is_list, "insertion", options=ENUM_OPTIONS.get(name)))
    link_required = _type_links(category.ndx_type)
    for link_name, target_category in category.links.items():
        # An instance structurally needs its model: the converter fails to build an OpticalFiber/etc.
        # without one, even though the bare ndx link spec leaves the quantity unset. Treat it as required.
        required = link_name == "model" or link_required.get(link_name, False)
        specs.append(
            FieldSpec(
                link_name,
                required,
                f"Link to a defined {CATEGORIES[target_category].singular}.",
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
# Device serialization (one form entry <-> one ndx object dict)
# ----------------------------------------------------------------------------------------------------------------------
def _serialize_device(category: CategorySpec, entry: dict) -> dict:
    """Serialize one form entry into its ndx object dict (nesting the fiber_insertion group)."""
    obj: dict = {}
    insertion: dict = {}
    for spec in field_specs(category.key):
        value = entry.get(spec.name)
        if spec.target == "insertion":
            if not _is_empty(value):
                insertion[spec.name] = value
        elif not _is_empty(value):
            obj[spec.name] = value
    if category.has_insertion:
        # Mandatory group (quantity 1): always present, even if the user left the coordinates blank.
        obj["fiber_insertion"] = insertion
    return obj


def _deserialize_device(category: CategorySpec, object_dict: dict) -> dict:
    """Flatten one ndx object dict back into a form entry (un-nesting fiber_insertion)."""
    entry = {key: value for key, value in object_dict.items() if key != "fiber_insertion"}
    entry.update(object_dict.get("fiber_insertion") or {})
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
        objects = [
            _serialize_device(category, entry)
            for entry in devices.get(category.key, [])
            if not _is_empty(entry.get("name"))
        ]
        if objects:
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
        objects = fiber_photometry.get(category.list_key, [])
        devices[category.key] = [_deserialize_device(category, obj) for obj in objects]

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
# Validation (fail early, before the YAML reaches the converter)
# ----------------------------------------------------------------------------------------------------------------------
# NWBFile/Subject scalars the converter genuinely needs to write a valid file.
_REQUIRED_NWBFILE_KEYS = ("session_description",)
_REQUIRED_SUBJECT_KEYS = ("subject_id", "sex", "species")


def validate_metadata_dict(metadata: dict, channels: list[Channel]) -> list[str]:
    """Return a list of human-readable problems that would break NWB export.

    Checks the metadata the form would export: required scalars, required device
    fields and links, referential integrity of every link (it must name a device
    that is actually defined), and per-channel required wavelengths and links. An
    empty list means the metadata is complete enough to export.
    """
    errors: list[str] = []

    nwbfile = metadata.get("NWBFile", {})
    for key in _REQUIRED_NWBFILE_KEYS:
        if _is_empty(nwbfile.get(key)):
            errors.append(f"NWBFile.{key} is required.")
    subject = metadata.get("Subject", {})
    for key in _REQUIRED_SUBJECT_KEYS:
        if _is_empty(subject.get(key)):
            errors.append(f"Subject.{key} is required.")

    fiber_photometry = metadata.get("Ophys", {}).get("FiberPhotometry", {})
    defined_names: dict[str, set[str]] = {
        category.key: {obj.get("name") for obj in fiber_photometry.get(category.list_key, [])}
        for category in CATEGORIES.values()
    }

    for category in CATEGORIES.values():
        for obj in fiber_photometry.get(category.list_key, []):
            label = f"{category.singular} '{obj.get('name') or '(unnamed)'}'"
            for spec in field_specs(category.key):
                if spec.target == "insertion":
                    continue
                value = obj.get(spec.name)
                if spec.link_target:
                    if spec.required and _is_empty(value):
                        errors.append(f"{label}: {spec.name} is required.")
                    elif not _is_empty(value) and value not in defined_names[spec.link_target]:
                        errors.append(
                            f"{label}: {spec.name} '{value}' is not a defined "
                            f"{CATEGORIES[spec.link_target].singular}."
                        )
                elif spec.required and _is_empty(value):
                    errors.append(f"{label}: {spec.name} is required.")

    rows = fiber_photometry.get("FiberPhotometryTable", {}).get("rows", [])
    for index, channel in enumerate(channels):
        row = rows[index] if index < len(rows) else {}
        label = f"channel {channel.region}/{channel.role}"
        for wavelength in CHANNEL_WAVELENGTHS:
            if _is_empty(row.get(wavelength)):
                errors.append(f"{label}: {wavelength} is required.")
        for link_name, target in CHANNEL_LINKS.items():
            value = row.get(link_name)
            if link_name in CHANNEL_REQUIRED_LINKS and _is_empty(value):
                errors.append(f"{label}: {link_name} link is required.")
            elif not _is_empty(value) and value not in defined_names[target]:
                errors.append(f"{label}: {link_name} '{value}' is not a defined {CATEGORIES[target].singular}.")

    return errors


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
