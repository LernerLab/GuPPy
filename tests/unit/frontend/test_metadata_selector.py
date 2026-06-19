"""Unit tests for the channel-centric NWB metadata form widget.

``MetadataSelector`` is a Panel widget, but it is fully driveable in-process: its
values can be set in Python and its button callbacks fired synchronously
(``button.clicks += 1``), so the form logic is tested directly without a browser.
"""

import pytest

from guppy.frontend.metadata_selector import MetadataSelector, _as_float, _read_widget
from guppy.utils.nwb_metadata import (
    Channel,
    FieldSpec,
    build_metadata_dict,
    dumps_yaml,
)

CHANNELS = [Channel("dms", "control", "Dv1A"), Channel("dms", "signal", "Dv2A")]

# A complete device library + channel annotations that pass validation, reused by the
# round-trip tests below.
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
COMPLETE_SCALARS = {
    "session_description": "RI30",
    "subject_id": "63",
    "sex": "M",
    "species": "Mus musculus",
    "experimenter": ["Doe, Jane"],
    "age": "P90D",
}


class TestMetadataSelector:
    @pytest.fixture
    def selector(self, panel_extension) -> MetadataSelector:
        return MetadataSelector(session_label="Photo (run1)", channels=CHANNELS, initial_metadata={})

    # -- Experimenter repeater -----------------------------------------------------------------------------------------
    def test_starts_with_one_blank_experimenter(self, selector):
        assert len(selector.experimenter_inputs) == 1
        assert selector.get_scalars()["experimenter"] == []

    def test_add_and_read_experimenters_strips_blanks(self, selector):
        selector.experimenter_inputs[0].value = "Doe, Jane"
        selector._add_experimenter("Smith, Joe")
        selector._add_experimenter("   ")  # blank -> dropped on read
        assert len(selector.experimenter_inputs) == 3
        assert selector.get_scalars()["experimenter"] == ["Doe, Jane", "Smith, Joe"]

    def test_remove_experimenter(self, selector):
        selector._add_experimenter("Smith, Joe")
        row = selector.experimenter_box[-1]
        text_input = selector.experimenter_inputs[-1]
        selector._remove_experimenter(row, text_input)
        assert text_input not in selector.experimenter_inputs
        assert row not in selector.experimenter_box

    # -- Age vs date-of-birth toggle ----------------------------------------------------------------------------------
    def test_age_branch_blanks_date_of_birth(self, selector):
        selector.age.value = "P90D"
        selector.date_of_birth.value = "2020-01-01"
        selector.age_or_dob.value = "Age"
        scalars = selector.get_scalars()
        assert scalars["age"] == "P90D"
        assert scalars["date_of_birth"] == ""

    def test_date_of_birth_branch_blanks_age(self, selector):
        selector.age.value = "P90D"
        selector.date_of_birth.value = "2020-01-01"
        selector.age_or_dob.value = "Date of birth"
        scalars = selector.get_scalars()
        assert scalars["date_of_birth"] == "2020-01-01"
        assert scalars["age"] == ""

    def test_sync_age_dob_toggles_visibility(self, selector):
        selector.age_or_dob.value = "Age"
        assert selector.age.visible is True
        assert selector.date_of_birth.visible is False
        selector.age_or_dob.value = "Date of birth"
        assert selector.age.visible is False
        assert selector.date_of_birth.visible is True

    # -- Device library ------------------------------------------------------------------------------------------------
    def test_add_device_appends_record_and_retitles(self, selector):
        selector._add_device("optical_fiber_model", {"name": "fmodel", "numerical_aperture": 0.48})
        records = selector.device_entry_records["optical_fiber_model"]
        assert len(records) == 1
        assert selector._entry_name(records[0]) == "fmodel"
        assert selector.category_cards["optical_fiber_model"].title.endswith("·  1")

    def test_get_devices_returns_only_named_entries(self, selector):
        selector._add_device("optical_fiber_model", {"name": "fmodel", "numerical_aperture": 0.48})
        selector._add_device("optical_fiber_model", {})  # unnamed -> excluded from read-back
        devices = selector.get_devices()
        assert devices["optical_fiber_model"] == [{"name": "fmodel", "numerical_aperture": 0.48}]

    def test_remove_device(self, selector):
        selector._add_device("optical_fiber_model", {"name": "fmodel"})
        record = selector.device_entry_records["optical_fiber_model"][0]
        selector._remove_device("optical_fiber_model", record)
        assert selector.device_entry_records["optical_fiber_model"] == []
        assert selector.category_cards["optical_fiber_model"].title == "Optical fiber models"

    def test_device_name_change_retitles_card(self, selector):
        selector._add_device("optical_fiber_model", {"name": "old"})
        record = selector.device_entry_records["optical_fiber_model"][0]
        name_widget = next(widget for spec, widget in record["fields"] if spec.name == "name")
        name_widget.value = "new"
        assert record["container"].title == "new"

    # -- Link-dropdown refresh -----------------------------------------------------------------------------------------
    def test_refresh_link_options_required_omits_empty_optional_keeps_it(self, selector):
        selector._add_device("optical_fiber", {"name": "fiber"})
        selector.refresh_link_options()
        # optical_fiber is a required channel link -> no empty option once a device exists.
        assert selector.channel_records[0]["fields"]["optical_fiber"].options == ["fiber"]
        # excitation_filter is optional and has no device of that category -> only the empty option.
        assert selector.channel_records[0]["fields"]["excitation_filter"].options == [""]

    def test_refresh_link_options_falls_back_when_current_value_disappears(self, selector):
        selector._add_device("optical_fiber", {"name": "fiber"})
        selector.refresh_link_options()
        select = selector.channel_records[0]["fields"]["optical_fiber"]
        select.value = "fiber"
        record = selector.device_entry_records["optical_fiber"][0]
        selector._remove_device("optical_fiber", record)  # refreshes internally
        assert select.options == [""]
        assert select.value == ""

    # -- Channel rows --------------------------------------------------------------------------------------------------
    def test_get_channel_rows_reads_wavelengths_and_links(self, selector):
        selector._add_device("optical_fiber", {"name": "fiber"})
        selector.refresh_link_options()
        fields = selector.channel_records[0]["fields"]
        fields["excitation_wavelength_in_nm"].value = 405.0
        fields["emission_wavelength_in_nm"].value = 525.0
        fields["optical_fiber"].value = "fiber"
        row = selector.get_channel_rows()[0]
        assert row["excitation_wavelength_in_nm"] == 405.0
        assert row["emission_wavelength_in_nm"] == 525.0
        assert row["optical_fiber"] == "fiber"

    # -- Full form round trip ------------------------------------------------------------------------------------------
    def test_set_from_metadata_round_trips_through_read_back(self, selector):
        built = build_metadata_dict(COMPLETE_DEVICES, COMPLETE_ROWS, COMPLETE_SCALARS, CHANNELS)
        selector.set_from_metadata(built)

        scalars = selector.get_scalars()
        assert scalars["session_description"] == "RI30"
        assert scalars["subject_id"] == "63"
        assert scalars["sex"] == "M"
        assert scalars["species"] == "Mus musculus"
        assert scalars["experimenter"] == ["Doe, Jane"]
        assert scalars["age"] == "P90D"

        # Rebuilding from the form's read-back must reproduce the original metadata exactly.
        rebuilt = build_metadata_dict(
            selector.get_devices(), selector.get_channel_rows(), selector.get_scalars(), CHANNELS
        )
        assert rebuilt == built

    # -- YAML editor + alert + path -----------------------------------------------------------------------------------
    def test_yaml_editor_round_trip(self, selector):
        metadata = {"NWBFile": {"lab": "Lerner"}}
        selector.set_yaml(metadata)
        assert selector.get_yaml() == metadata

    def test_set_alert_message_colors_errors_red(self, selector):
        selector.set_alert_message("####Alert !! \n something missing")
        assert selector.alert.alert_type == "danger"

    def test_set_alert_message_colors_success_green(self, selector):
        selector.set_alert_message("#### No alerts !!")
        assert selector.alert.alert_type == "success"

    def test_set_path_updates_widget(self, selector):
        selector.set_path("/tmp/nwb_metadata.yaml")
        assert selector.path.value == "/tmp/nwb_metadata.yaml"

    # -- File upload ---------------------------------------------------------------------------------------------------
    def test_on_file_upload_populates_form(self, selector):
        built = build_metadata_dict(COMPLETE_DEVICES, COMPLETE_ROWS, COMPLETE_SCALARS, CHANNELS)
        selector.load_existing.filename = "other_session.yaml"
        # Setting the FileInput value fires the registered _on_file_upload watcher.
        selector.load_existing.value = dumps_yaml(built).encode("utf-8")
        assert "Loaded metadata from other_session.yaml" in selector.alert.object
        assert selector.get_scalars()["session_description"] == "RI30"

    def test_on_file_upload_reports_invalid_yaml(self, selector):
        selector.load_existing.value = b"*undefined_alias"
        assert "Could not read uploaded YAML" in selector.alert.object
        assert selector.alert.alert_type == "danger"

    def test_on_file_upload_no_value_is_noop(self, selector):
        before = selector.alert.object
        selector.load_existing.value = b""  # falsy -> early return, nothing changes
        selector._on_file_upload(event=None)
        assert selector.alert.object == before

    # -- Callback wiring -----------------------------------------------------------------------------------------------
    def test_attach_callbacks_fires_on_button_click(self, selector):
        calls = []
        selector.attach_callbacks(
            {
                "build_config": lambda event: calls.append("build"),
                "save": lambda event: calls.append("save"),
            }
        )
        selector.build_config.clicks += 1
        selector.save.clicks += 1
        assert calls == ["build", "save"]


class TestAsFloat:
    def test_none_and_blank_return_none(self):
        assert _as_float(None) is None
        assert _as_float("") is None
        assert _as_float("   ") is None

    def test_coerces_scalar(self):
        assert _as_float("3.5") == 3.5
        assert _as_float(2) == 2.0

    def test_list_index_extraction(self):
        assert _as_float([1.0, 2.0], 0) == 1.0
        assert _as_float([1.0, 2.0], 1) == 2.0

    def test_index_out_of_range_or_not_a_list_returns_none(self):
        assert _as_float([1.0], 1) is None
        assert _as_float("not-a-list", 0) is None


class TestReadWidget:
    def test_link_target_returns_value_or_none(self):
        spec = FieldSpec("model", True, "doc", "text", False, "self", link_target="optical_fiber_model")
        assert _read_widget(spec, _StubWidget(value="fmodel")) == "fmodel"
        assert _read_widget(spec, _StubWidget(value="")) is None

    def test_list_pair_returns_floats_or_none(self):
        spec = FieldSpec("location", False, "doc", "float", True, "self")
        present = _StubRow(_StubWidget(value=1.0), _StubWidget(value=2.0))
        assert _read_widget(spec, present) == [1.0, 2.0]
        missing = _StubRow(_StubWidget(value=None), _StubWidget(value=2.0))
        assert _read_widget(spec, missing) is None

    def test_float_returns_raw(self):
        spec = FieldSpec("numerical_aperture", True, "doc", "float", False, "self")
        assert _read_widget(spec, _StubWidget(value=0.48)) == 0.48

    def test_text_blank_returns_none(self):
        spec = FieldSpec("serial_number", False, "doc", "text", False, "self")
        assert _read_widget(spec, _StubWidget(value="OF1")) == "OF1"
        assert _read_widget(spec, _StubWidget(value="  ")) is None


class _StubWidget:
    """A minimal stand-in for a Panel input widget (only ``.value`` is read)."""

    def __init__(self, value):
        self.value = value


class _StubRow:
    """A stand-in for the (min, max) FloatInput pair row read via ``_field_pair``."""

    def __init__(self, low, high):
        self._field_pair = (low, high)
