import panel as pn
import pytest

from guppy.frontend.store_labeling_config import (
    NAME_OR_SIGNAL_HELP,
    NO_SIGNAL_OPTION,
    StoreLabelingConfig,
)


@pytest.fixture
def show_config_button():
    return pn.widgets.Button(name="Show Selected Configuration", width=600)


def build_config(show_config_button, store_ids, store_id_to_store_labels=None):
    """Construct a StoreLabelingConfig and return it plus its widget dicts."""
    dropdowns, textboxes, control_refs = {}, {}, {}
    config = StoreLabelingConfig(
        show_config_button=show_config_button,
        store_id_dropdowns=dropdowns,
        store_id_textboxes=textboxes,
        store_id_control_refs=control_refs,
        store_ids=store_ids,
        store_id_to_store_labels=store_id_to_store_labels or {},
    )
    return config, dropdowns, textboxes, control_refs


@pytest.fixture
def store_labeling_config_instance(panel_extension, show_config_button):
    config, _, _, _ = build_config(show_config_button, ["Dv1A"])
    return config


class TestStoreLabelingConfig:
    # ── Empty store_ids ──────────────────────────────────────────────────────

    def test_empty_store_ids_yields_no_widgets(self, panel_extension, show_config_button):
        config, _, _, _ = build_config(show_config_button, [])
        assert config.config_widgets == []

    # ── Single store_id, no cache ────────────────────────────────────────────

    def test_single_store_id_widget_count(self, panel_extension, show_config_button):
        # intro markdown + column heading + store rows + spacer + show button + instructions markdown = 6
        config, _, _, _ = build_config(show_config_button, ["Dv1A"])
        assert len(config.config_widgets) == 6

    def test_each_store_is_one_row_of_equal_width(self, panel_extension, show_config_button):
        config, dropdowns, _, _ = build_config(show_config_button, ["Dv1A", "Dv2A", "PrtN"])

        store_entries = config.config_widgets[2]
        assert type(store_entries) is pn.Column
        assert [type(entry) for entry in store_entries] == [pn.Row, pn.Row, pn.Row]
        assert [entry.width for entry in store_entries] == [630, 630, 630]
        assert [entry[1] for entry in store_entries] == [
            dropdowns["Dv1A_0"],
            dropdowns["Dv2A_1"],
            dropdowns["PrtN_2"],
        ]

    def test_store_id_is_a_semibold_title(self, panel_extension, show_config_button):
        config, _, _, _ = build_config(show_config_button, ["Dv1A"])

        store_label = config.config_widgets[2][0][0]
        assert store_label.object == "<div style='font-size: 14px; font-weight: 600; padding-top: 7px;'>Dv1A</div>"

    def test_column_heading_explains_the_name_column_in_a_tooltip(self, panel_extension, show_config_button):
        config, _, _, _ = build_config(show_config_button, ["Dv1A"])

        heading = config.config_widgets[1]
        style = "font-size: 12px; font-weight: 600; color: #6C757D;"
        assert heading[0].object == f"<div style='{style}'>Store</div>"
        assert heading[1].object == f"<div style='{style}'>Type</div>"
        assert heading[2][0].object == f"<div style='{style}'>Name or Signal</div>"
        assert isinstance(heading[2][1], pn.widgets.TooltipIcon)
        assert heading[2][1].value == NAME_OR_SIGNAL_HELP

    def test_instructions_are_small_text(self, panel_extension, show_config_button):
        config, _, _, _ = build_config(show_config_button, ["Dv1A"])

        instructions = config.config_widgets[-1]
        assert instructions.object == "Click Show Selected Configuration to apply your selections."
        assert instructions.css_classes == ["small"]

    def test_single_store_id_dropdown_in_dict(self, panel_extension, show_config_button):
        _, dropdowns, _, _ = build_config(show_config_button, ["Dv1A"])
        assert "Dv1A_0" in dropdowns
        assert dropdowns["Dv1A_0"].value == ""

    def test_single_store_id_textbox_in_dict(self, panel_extension, show_config_button):
        _, _, textboxes, _ = build_config(show_config_button, ["Dv1A"])
        assert "Dv1A_0" in textboxes
        assert textboxes["Dv1A_0"].value == ""

    def test_single_store_id_control_ref_in_dict(self, panel_extension, show_config_button):
        _, _, _, control_refs = build_config(show_config_button, ["Dv1A"])
        assert "Dv1A_0" in control_refs
        assert control_refs["Dv1A_0"].value == ""

    # ── _parse_cached_value ───────────────────────────────────────────────────

    def test_parse_cached_value_control_prefix(self, store_labeling_config_instance):
        assert store_labeling_config_instance._parse_cached_value("control_DMS") == ("control", "DMS")

    def test_parse_cached_value_signal_prefix(self, store_labeling_config_instance):
        assert store_labeling_config_instance._parse_cached_value("signal_DLS") == ("signal", "DLS")

    def test_parse_cached_value_no_prefix_treated_as_event(self, store_labeling_config_instance):
        assert store_labeling_config_instance._parse_cached_value("RewardedPort") == ("event TTLs", "RewardedPort")

    def test_parse_cached_value_empty_string(self, store_labeling_config_instance):
        assert store_labeling_config_instance._parse_cached_value("") == ("", "")

    # ── _name_placeholder ─────────────────────────────────────────────────────

    def test_name_placeholder_signal(self, store_labeling_config_instance):
        assert store_labeling_config_instance._name_placeholder("signal") == "Enter the recording site"

    def test_name_placeholder_event_ttls(self, store_labeling_config_instance):
        assert store_labeling_config_instance._name_placeholder("event TTLs") == "Enter the event name"

    def test_name_placeholder_covariate(self, store_labeling_config_instance):
        assert store_labeling_config_instance._name_placeholder("behavioral covariate") == "Enter the variable name"

    def test_name_placeholder_empty(self, store_labeling_config_instance):
        assert store_labeling_config_instance._name_placeholder("") == ""

    # ── Cache pre-population ──────────────────────────────────────────────────

    def test_cache_control_pre_populates_dropdown(self, panel_extension, show_config_button):
        _, dropdowns, textboxes, _ = build_config(show_config_button, ["Dv1A"], {"Dv1A": ["control_DMS"]})
        assert dropdowns["Dv1A_0"].value == "control"
        # The control's own Name box is unused (it references a signal), but the cached
        # recording site is remembered so it can re-select the signal once one is defined.
        assert textboxes["Dv1A_0"].value == "DMS"

    def test_cache_signal_pre_populates_dropdown_and_textbox(self, panel_extension, show_config_button):
        _, dropdowns, textboxes, _ = build_config(show_config_button, ["Dv2A"], {"Dv2A": ["signal_DLS"]})
        assert dropdowns["Dv2A_0"].value == "signal"
        assert textboxes["Dv2A_0"].value == "DLS"

    def test_cache_event_ttl_pre_populates_dropdown_and_textbox(self, panel_extension, show_config_button):
        _, dropdowns, textboxes, _ = build_config(show_config_button, ["PrtR"], {"PrtR": ["RewardedPort"]})
        assert dropdowns["PrtR_0"].value == "event TTLs"
        assert textboxes["PrtR_0"].value == "RewardedPort"

    def test_cache_control_reselects_its_signal_when_available(self, panel_extension, show_config_button):
        # Dv2A is the signal "DMS"; Dv1A is its cached control → the control-for
        # dropdown should resolve back to Dv2A's widget key.
        _, dropdowns, _, control_refs = build_config(
            show_config_button,
            ["Dv1A", "Dv2A"],
            {"Dv1A": ["control_DMS"], "Dv2A": ["signal_DMS"]},
        )
        assert control_refs["Dv1A_0"].value == "Dv2A_1"

    # ── Dropdown change updates placeholder + visibility ──────────────────────

    def test_cached_type_sets_the_name_placeholder(self, panel_extension, show_config_button):
        _, _, textboxes, _ = build_config(show_config_button, ["Dv2A"], {"Dv2A": ["signal_DLS"]})
        assert textboxes["Dv2A_0"].placeholder == "Enter the recording site"

    def test_dropdown_change_to_event_updates_placeholder(self, panel_extension, show_config_button):
        _, dropdowns, textboxes, _ = build_config(show_config_button, ["PrtR"])
        dropdowns["PrtR_0"].value = "event TTLs"
        assert textboxes["PrtR_0"].placeholder == "Enter the event name"

    def test_control_row_shows_signal_picker_hides_name_box(self, panel_extension, show_config_button):
        _, dropdowns, textboxes, control_refs = build_config(show_config_button, ["Dv1A"])
        dropdowns["Dv1A_0"].value = "control"
        assert control_refs["Dv1A_0"].visible is True
        assert textboxes["Dv1A_0"].visible is False

    def test_signal_row_shows_name_box_hides_signal_picker(self, panel_extension, show_config_button):
        _, dropdowns, textboxes, control_refs = build_config(show_config_button, ["Dv2A"])
        dropdowns["Dv2A_0"].value = "signal"
        assert textboxes["Dv2A_0"].visible is True
        assert control_refs["Dv2A_0"].visible is False

    # ── Reactive control-for options ──────────────────────────────────────────

    def test_control_options_track_signal_rows(self, panel_extension, show_config_button):
        _, dropdowns, textboxes, control_refs = build_config(show_config_button, ["Dv1A", "Dv2A"])
        # Mark Dv2A a signal named "left_hemisphere" (with an underscore).
        dropdowns["Dv2A_1"].value = "signal"
        textboxes["Dv2A_1"].value = "left_hemisphere"
        # Dv1A's control-for dropdown should now offer that signal.
        options = control_refs["Dv1A_0"].options
        assert options["Dv2A – left_hemisphere"] == "Dv2A_1"
        assert NO_SIGNAL_OPTION in options

    # ── Behavioral covariate rows ─────────────────────────────────────────────

    def test_behavioral_covariate_is_offered(self, panel_extension, show_config_button):
        _, dropdowns, _, _ = build_config(show_config_button, ["akinesia"])

        assert dropdowns["akinesia_0"].options == ["", "control", "signal", "event TTLs", "behavioral covariate"]

    def test_covariate_row_shows_name_box_hides_signal_picker(self, panel_extension, show_config_button):
        _, dropdowns, textboxes, control_refs = build_config(show_config_button, ["akinesia"])

        dropdowns["akinesia_0"].value = "behavioral covariate"

        assert textboxes["akinesia_0"].visible is True
        assert control_refs["akinesia_0"].visible is False

    def test_cached_covariate_repopulates_as_a_covariate(self, panel_extension, show_config_button):
        # Regression guard: without its own branch the cache round-trip falls through
        # to the event catch-all and comes back named "covariate_akinesia".
        _, dropdowns, textboxes, _ = build_config(
            show_config_button, ["akinesia"], {"akinesia": ["covariate_akinesia"]}
        )

        assert dropdowns["akinesia_0"].value == "behavioral covariate"
        assert textboxes["akinesia_0"].value == "akinesia"
