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
  rows and the per-role response-series regions from the channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from functools import cache
from pathlib import Path

import numpy as np
import yaml

from .stores_list import read_stores_list
from ..analysis.io_utils import (
    CONTROL_PREFIX,
    SIGNAL_PREFIX,
    get_control_and_signal_channel_names,
    recording_site_from_channel_label,
)


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

    recording_site: str
    role: str  # "signal" or "control"
    store_name: str

    @property
    def table_row_key(self) -> str:
        """Key of this channel's FiberPhotometryTable row (its metadata_key handle)."""
        return f"{self.recording_site}_{self.role}"


def derive_channels(*, output_dir: str | Path) -> list[Channel]:
    """Return the ordered fiber-photometry channels for a GuPPy output directory.

    Reads ``<output_dir>/storesList.csv`` and pairs each ``signal_<recording_site>`` label
    with its ``control_<recording_site>``. Recording sites run in ``storesList.csv`` column
    order and the two roles of a site are adjacent, control first.

    The column order matters: the converter walks recording sites the same way and zips each
    response series' ``fiber_photometry_table_region`` against that order to recover which
    fiber recorded which site.
    """
    stores_list = read_stores_list(run_folder=output_dir)
    store_labels = [str(label) for label in stores_list[1, :]]
    label_to_store_name = {label: str(store) for store, label in zip(stores_list[0, :], store_labels, strict=True)}

    # Called for its validation: it raises when a signal has no matching control (or vice versa).
    # Its own output is sorted by recording site, which is why the order is re-derived below.
    get_control_and_signal_channel_names(stores_list)

    control_label_by_recording_site = {
        recording_site_from_channel_label(label): label
        for label in store_labels
        if label.lower().startswith(CONTROL_PREFIX)
    }
    channels: list[Channel] = []
    for label in store_labels:
        if not label.lower().startswith(SIGNAL_PREFIX):
            continue
        recording_site = recording_site_from_channel_label(label)
        control_label = control_label_by_recording_site[recording_site]
        channels.append(
            Channel(recording_site=recording_site, role="control", store_name=label_to_store_name[control_label])
        )
        channels.append(Channel(recording_site=recording_site, role="signal", store_name=label_to_store_name[label]))
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


# The shared, cross-modality device registries. They sit at the top level of the metadata dict
# rather than under FiberPhotometry, because ndx-ophys-devices classes are reused by other
# modalities. Their entries carry a ``type`` naming the concrete ndx class to build.
DEVICE_MODELS_KEY = "DeviceModels"
DEVICES_KEY = "Devices"
FIBER_PHOTOMETRY_KEY = "FiberPhotometry"


@dataclass(frozen=True)
class CategorySpec:
    """A user-facing device category and where its entries live in the metadata dict."""

    key: str
    label: str  # plural section title
    singular: str  # used for the "Add <singular>" button
    ndx_type: str  # installed type to introspect for fields/docs/required
    destination: tuple[str, ...]  # path of the dict holding this category's entries
    has_insertion: bool = False  # OpticalFiber: emit a (mandatory) fiber_insertion group
    skip_deprecated: bool = False  # device instances: drop the deprecated model-ish attrs
    links: dict[str, str] = field(default_factory=dict)  # link field name -> target category key


# Ordered by dependency: a model before the instance that links to it; viral vector before its
# injection before the indicator that references the injection. Instances are what channels link to.
CATEGORIES: dict[str, CategorySpec] = {
    "optical_fiber_model": CategorySpec(
        "optical_fiber_model",
        "Optical fiber models",
        "optical fiber model",
        "OpticalFiberModel",
        (DEVICE_MODELS_KEY,),
    ),
    "optical_fiber": CategorySpec(
        "optical_fiber",
        "Optical fibers",
        "optical fiber",
        "OpticalFiber",
        (DEVICES_KEY,),
        has_insertion=True,
        skip_deprecated=True,
        links={"model": "optical_fiber_model"},
    ),
    "excitation_source_model": CategorySpec(
        "excitation_source_model",
        "Excitation source models",
        "excitation source model",
        "ExcitationSourceModel",
        (DEVICE_MODELS_KEY,),
    ),
    "excitation_source": CategorySpec(
        "excitation_source",
        "Excitation sources",
        "excitation source",
        "ExcitationSource",
        (DEVICES_KEY,),
        skip_deprecated=True,
        links={"model": "excitation_source_model"},
    ),
    "photodetector_model": CategorySpec(
        "photodetector_model",
        "Photodetector models",
        "photodetector model",
        "PhotodetectorModel",
        (DEVICE_MODELS_KEY,),
    ),
    "photodetector": CategorySpec(
        "photodetector",
        "Photodetectors",
        "photodetector",
        "Photodetector",
        (DEVICES_KEY,),
        skip_deprecated=True,
        links={"model": "photodetector_model"},
    ),
    "band_optical_filter_model": CategorySpec(
        "band_optical_filter_model",
        "Band optical filter models",
        "band optical filter model",
        "BandOpticalFilterModel",
        (DEVICE_MODELS_KEY,),
    ),
    "band_optical_filter": CategorySpec(
        "band_optical_filter",
        "Band optical filters",
        "band optical filter",
        "BandOpticalFilter",
        (DEVICES_KEY,),
        skip_deprecated=True,
        links={"model": "band_optical_filter_model"},
    ),
    "dichroic_mirror_model": CategorySpec(
        "dichroic_mirror_model",
        "Dichroic mirror models",
        "dichroic mirror model",
        "DichroicMirrorModel",
        (DEVICE_MODELS_KEY,),
    ),
    "dichroic_mirror": CategorySpec(
        "dichroic_mirror",
        "Dichroic mirrors",
        "dichroic mirror",
        "DichroicMirror",
        (DEVICES_KEY,),
        skip_deprecated=True,
        links={"model": "dichroic_mirror_model"},
    ),
    "virus": CategorySpec("virus", "Viruses", "virus", "ViralVector", (FIBER_PHOTOMETRY_KEY, "FiberPhotometryViruses")),
    "virus_injection": CategorySpec(
        "virus_injection",
        "Virus injections",
        "virus injection",
        "ViralVectorInjection",
        (FIBER_PHOTOMETRY_KEY, "FiberPhotometryVirusInjections"),
        links={"viral_vector": "virus"},
    ),
    "indicator": CategorySpec(
        "indicator",
        "Indicators",
        "indicator",
        "Indicator",
        (FIBER_PHOTOMETRY_KEY, "FiberPhotometryIndicators"),
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

# Entries reference each other by metadata_key, under a field named for the *kind* of thing
# referenced. Every link field is its spec link name plus a ``_metadata_key`` suffix, except an
# ndx device's ``model`` link, which the shared device registry calls ``device_model``.
_LINK_FIELD_OVERRIDES = {"model": "device_model_metadata_key"}


def link_field_name(link_name: str) -> str:
    """Return the metadata field a link is written under (the form calls it ``link_name``)."""
    return _LINK_FIELD_OVERRIDES.get(link_name, f"{link_name}_metadata_key")


def _registry_type(category: CategorySpec) -> str | None:
    """Return the ``type`` a category's entries carry, or None for the type-homogeneous collections.

    The shared device registries mix categories, so each entry names its concrete ndx class; the
    fiber-photometry collections hold one type each and carry no discriminator.
    """
    if category.destination[0] in (DEVICES_KEY, DEVICE_MODELS_KEY):
        return category.ndx_type
    return None


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


@cache
def _type_spec(type_name: str):  # noqa: ANN202  (returns an hdmf Spec)
    import ndx_fiber_photometry  # noqa: F401  (register extensions)
    import ndx_ophys_devices  # noqa: F401
    from pynwb import get_type_map

    namespaces = ("ndx-ophys-devices", "ndx-fiber-photometry")
    catalog = get_type_map().namespace_catalog
    for namespace in namespaces:
        try:
            return catalog.get_spec(namespace, type_name)
        except ValueError:
            continue
    raise ValueError(
        f"Type {type_name!r} is not defined by any of the installed extensions {namespaces}. The "
        f"metadata form is built from the installed specs, so this means the installed "
        f"ndx-ophys-devices/ndx-fiber-photometry versions are older than GuPPy expects; upgrade them."
    )


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
NWBFILE_KEYS = ("session_description", "identifier", "session_start_time", "lab", "institution", "experimenter")
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


def parse_session_start_time(value: object) -> datetime | None:
    """Return an ISO 8601 session start time as a ``datetime``, or ``None`` when empty.

    The value must reach the converter as a real timestamp: a quoted string in the YAML would come
    back from ``yaml.safe_load`` as a ``str``, which ``NWBFile`` rejects.

    Raises
    ------
    ValueError
        If ``value`` is neither empty nor a valid ISO 8601 timestamp.
    """
    if _is_empty(value):
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).strip())


def format_session_start_time(value: object) -> str:
    """Return a session start time as the ISO 8601 string the form edits, or ``""`` when empty."""
    if _is_empty(value):
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


# ----------------------------------------------------------------------------------------------------------------------
# Device serialization (one form entry <-> one ndx object dict)
# ----------------------------------------------------------------------------------------------------------------------
def _serialize_device(*, category: CategorySpec, entry: dict) -> dict:
    """Serialize one form entry into its ndx object dict (nesting the fiber_insertion group)."""
    obj: dict = {}
    registry_type = _registry_type(category)
    if registry_type is not None:
        obj["type"] = registry_type
    insertion: dict = {}
    for spec in field_specs(category.key):
        value = entry.get(spec.name)
        if spec.target == "insertion":
            if not _is_empty(value):
                insertion[spec.name] = value
        elif not _is_empty(value):
            obj[link_field_name(spec.name) if spec.link_target else spec.name] = value
    if category.has_insertion:
        # Mandatory group (quantity 1): always present, even if the user left the coordinates blank.
        obj["fiber_insertion"] = insertion
    return obj


def _deserialize_device(*, category: CategorySpec, object_dict: dict) -> dict:
    """Flatten one ndx object dict back into a form entry (un-nesting fiber_insertion)."""
    entry = {key: value for key, value in object_dict.items() if key not in ("fiber_insertion", "type")}
    entry.update(object_dict.get("fiber_insertion") or {})
    for link_name in category.links:
        field_name = link_field_name(link_name)
        if field_name in entry:
            entry[link_name] = entry.pop(field_name)
    return entry


def _collection(*, metadata: dict, destination: tuple[str, ...], create: bool) -> dict:
    """Return the dict at ``destination``, optionally creating the intermediate levels."""
    node = metadata
    for key in destination:
        if create:
            node = node.setdefault(key, {})
        else:
            node = node.get(key, {})
    return node


# ----------------------------------------------------------------------------------------------------------------------
# build / parse
# ----------------------------------------------------------------------------------------------------------------------
_ROLE_SERIES_DESCRIPTIONS = {
    "signal": "Calcium-dependent fluorescence, one column per recording site.",
    "control": "Isosbestic control fluorescence, one column per recording site.",
}


def build_metadata_dict(
    *, devices: dict[str, list[dict]], channel_rows: list[dict], scalars: dict, channels: list[Channel]
) -> dict:
    """Assemble the full session metadata overlay from the form state.

    Every entry is keyed by its user-entered ``name``, which doubles as its metadata_key --
    the handle other entries reference it by. Generates the FiberPhotometryTable rows (one per
    channel, ``location`` = recording site) and one response-series entry per role, each naming
    its rows in recording-site order.
    """
    metadata: dict = {}
    nwbfile = _drop_empty({key: scalars.get(key) for key in NWBFILE_KEYS})
    if "session_start_time" in nwbfile:
        nwbfile["session_start_time"] = parse_session_start_time(nwbfile["session_start_time"])
    subject = _drop_empty({key: scalars.get(key) for key in SUBJECT_KEYS})
    if nwbfile:
        metadata["NWBFile"] = nwbfile
    if subject:
        metadata["Subject"] = subject

    for category in CATEGORIES.values():
        entries = [entry for entry in devices.get(category.key, []) if not _is_empty(entry.get("name"))]
        if not entries:
            continue
        collection = _collection(metadata=metadata, destination=category.destination, create=True)
        for entry in entries:
            collection[entry["name"]] = _serialize_device(category=category, entry=entry)

    recording_sites = list(dict.fromkeys(channel.recording_site for channel in channels))
    table_rows: dict[str, dict] = {}
    for index, channel in enumerate(channels):
        row = {"location": channel.recording_site}
        annotations = channel_rows[index] if index < len(channel_rows) else {}
        row.update(_drop_empty({key: annotations.get(key) for key in CHANNEL_WAVELENGTHS}))
        row.update(_drop_empty({link_field_name(key): annotations.get(key) for key in CHANNEL_LINKS}))
        table_rows[channel.table_row_key] = row

    if table_rows:
        fiber_photometry = _collection(metadata=metadata, destination=(FIBER_PHOTOMETRY_KEY,), create=True)
        fiber_photometry["FiberPhotometryTable"] = {
            "name": scalars.get("fiber_photometry_table_name") or "fiber_photometry_table",
            "description": scalars.get("fiber_photometry_table_description") or "Fiber photometry table.",
            "rows": table_rows,
        }
        # One series per role, stacking that role's store from every recording site. The region
        # names one row per stacked column, so it must run in the same recording-site order.
        for role in dict.fromkeys(channel.role for channel in channels):
            fiber_photometry[role] = {
                "description": _ROLE_SERIES_DESCRIPTIONS[role],
                "fiber_photometry_table_region": [f"{site}_{role}" for site in recording_sites],
                "fiber_photometry_table_region_description": (
                    f"FiberPhotometryTable rows for the {role} channels, in recording-site order: "
                    f"{', '.join(recording_sites)}."
                ),
            }

    return metadata


def parse_metadata_dict(*, metadata: dict, channels: list[Channel]) -> tuple[dict[str, list[dict]], list[dict], dict]:
    """Inverse of :func:`build_metadata_dict` for repopulating the form from a saved dict."""
    devices: dict[str, list[dict]] = {}
    for category in CATEGORIES.values():
        collection = _collection(metadata=metadata, destination=category.destination, create=False)
        devices[category.key] = [
            _deserialize_device(category=category, object_dict=entry)
            for entry in collection.values()
            if entry.get("type") == _registry_type(category)
        ]

    table = metadata.get(FIBER_PHOTOMETRY_KEY, {}).get("FiberPhotometryTable", {})
    saved_rows = table.get("rows", {})
    channel_rows: list[dict] = []
    for channel in channels:
        row = saved_rows.get(channel.table_row_key, {})
        annotations = {key: row.get(key, "") for key in CHANNEL_WAVELENGTHS}
        annotations.update({key: row.get(link_field_name(key), "") for key in CHANNEL_LINKS})
        channel_rows.append(annotations)

    nwbfile = metadata.get("NWBFile", {})
    subject = metadata.get("Subject", {})
    scalars: dict = {key: nwbfile.get(key, "") for key in NWBFILE_KEYS}
    scalars["session_start_time"] = format_session_start_time(scalars["session_start_time"])
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


def validate_metadata_dict(
    *, metadata: dict, channels: list[Channel], require_session_start_time: bool = False
) -> list[str]:
    """Return a list of human-readable problems that would break NWB export.

    Checks the metadata the form would export: required scalars, required device
    fields and links, referential integrity of every link (it must name a device
    that is actually defined), and per-channel required wavelengths and links. An
    empty list means the metadata is complete enough to export.

    Parameters
    ----------
    metadata : dict
        The metadata overlay to check.
    channels : list of Channel
        The session's fiber-photometry channels, as derived from ``storesList.csv``.
    require_session_start_time : bool, optional
        Whether ``NWBFile.session_start_time`` must be supplied here. Set for the acquisition
        formats that do not record one, where the form is its only source. Default = False.
    """
    errors: list[str] = []

    nwbfile = metadata.get("NWBFile", {})
    for key in _REQUIRED_NWBFILE_KEYS:
        if _is_empty(nwbfile.get(key)):
            errors.append(f"NWBFile.{key} is required.")
    session_start_time = nwbfile.get("session_start_time")
    if require_session_start_time and _is_empty(session_start_time):
        errors.append("NWBFile.session_start_time is required: this session's acquisition format does not record one.")
    elif not _is_empty(session_start_time):
        try:
            parse_session_start_time(session_start_time)
        except ValueError:
            errors.append(f"NWBFile.session_start_time '{session_start_time}' is not a valid ISO 8601 timestamp.")
    subject = metadata.get("Subject", {})
    for key in _REQUIRED_SUBJECT_KEYS:
        if _is_empty(subject.get(key)):
            errors.append(f"Subject.{key} is required.")

    entries_by_category: dict[str, dict[str, dict]] = {}
    for category in CATEGORIES.values():
        collection = _collection(metadata=metadata, destination=category.destination, create=False)
        entries_by_category[category.key] = {
            key: entry for key, entry in collection.items() if entry.get("type") == _registry_type(category)
        }

    for category in CATEGORIES.values():
        for key, obj in entries_by_category[category.key].items():
            label = f"{category.singular} '{obj.get('name') or key}'"
            for spec in field_specs(category.key):
                if spec.target == "insertion":
                    continue
                if spec.link_target:
                    value = obj.get(link_field_name(spec.name))
                    if spec.required and _is_empty(value):
                        errors.append(f"{label}: {spec.name} is required.")
                    elif not _is_empty(value) and value not in entries_by_category[spec.link_target]:
                        errors.append(
                            f"{label}: {spec.name} '{value}' is not a defined "
                            f"{CATEGORIES[spec.link_target].singular}."
                        )
                elif spec.required and _is_empty(obj.get(spec.name)):
                    errors.append(f"{label}: {spec.name} is required.")

    rows = metadata.get(FIBER_PHOTOMETRY_KEY, {}).get("FiberPhotometryTable", {}).get("rows", {})
    for channel in channels:
        row = rows.get(channel.table_row_key, {})
        label = f"channel {channel.recording_site}/{channel.role}"
        for wavelength in CHANNEL_WAVELENGTHS:
            if _is_empty(row.get(wavelength)):
                errors.append(f"{label}: {wavelength} is required.")
        for link_name, target in CHANNEL_LINKS.items():
            value = row.get(link_field_name(link_name))
            if link_name in CHANNEL_REQUIRED_LINKS and _is_empty(value):
                errors.append(f"{label}: {link_name} link is required.")
            elif not _is_empty(value) and value not in entries_by_category[target]:
                errors.append(f"{label}: {link_name} '{value}' is not a defined {CATEGORIES[target].singular}.")

    return errors


# ----------------------------------------------------------------------------------------------------------------------
# YAML I/O
# ----------------------------------------------------------------------------------------------------------------------
def load_yaml(path: str | Path) -> dict:
    """Load a YAML file into a dict."""
    with open(path) as yaml_file:
        return yaml.safe_load(yaml_file) or {}


def dump_yaml(*, metadata: dict, path: str | Path) -> None:
    """Write ``metadata`` to ``path`` as human-editable YAML (insertion order preserved)."""
    with open(path, "w") as yaml_file:
        yaml.dump(metadata, yaml_file, Dumper=_RobustDumper, sort_keys=False, default_flow_style=False)


def loads_yaml(text: str) -> dict:
    """Parse a YAML string into a dict (blank text yields an empty dict)."""
    return yaml.safe_load(text) or {}


def dumps_yaml(metadata: dict) -> str:
    """Serialize ``metadata`` to a human-editable YAML string (insertion order preserved)."""
    return yaml.dump(metadata, Dumper=_RobustDumper, sort_keys=False, default_flow_style=False)
