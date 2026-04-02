import panel as pn
import pytest

from guppy.frontend.storenames_config import StorenamesConfig


@pytest.fixture
def show_config_button():
    return pn.widgets.Button(name="Show Selected Configuration", width=600)


@pytest.fixture
def storenames_config_instance(panel_extension, show_config_button):
    return StorenamesConfig(
        show_config_button=show_config_button,
        storename_dropdowns={},
        storename_textboxes={},
        storenames=["Dv1A"],
        storenames_cache={},
    )


class TestStorenamesConfig:
    # ── Empty storenames ──────────────────────────────────────────────────────

    def test_empty_storenames_yields_no_widgets(self, panel_extension, show_config_button):
        config = StorenamesConfig(
            show_config_button=show_config_button,
            storename_dropdowns={},
            storename_textboxes={},
            storenames=[],
            storenames_cache={},
        )
        assert config.config_widgets == []

    # ── Single storename, no cache ────────────────────────────────────────────

    def test_single_storename_widget_count(self, panel_extension, show_config_button):
        # markdown header + 1 row + spacer + show button + instructions markdown = 5
        config = StorenamesConfig(
            show_config_button=show_config_button,
            storename_dropdowns={},
            storename_textboxes={},
            storenames=["Dv1A"],
            storenames_cache={},
        )
        assert len(config.config_widgets) == 5

    def test_single_storename_dropdown_in_dict(self, panel_extension, show_config_button):
        dropdowns = {}
        StorenamesConfig(
            show_config_button=show_config_button,
            storename_dropdowns=dropdowns,
            storename_textboxes={},
            storenames=["Dv1A"],
            storenames_cache={},
        )
        assert "Dv1A_0" in dropdowns
        assert dropdowns["Dv1A_0"].value == ""

    def test_single_storename_textbox_in_dict(self, panel_extension, show_config_button):
        textboxes = {}
        StorenamesConfig(
            show_config_button=show_config_button,
            storename_dropdowns={},
            storename_textboxes=textboxes,
            storenames=["Dv1A"],
            storenames_cache={},
        )
        assert "Dv1A_0" in textboxes
        assert textboxes["Dv1A_0"].value == ""

    # ── _parse_cached_value ───────────────────────────────────────────────────

    def test_parse_cached_value_control_prefix(self, storenames_config_instance):
        result = storenames_config_instance._parse_cached_value("control_DMS")
        assert result == ("control", "DMS")

    def test_parse_cached_value_signal_prefix(self, storenames_config_instance):
        result = storenames_config_instance._parse_cached_value("signal_DLS")
        assert result == ("signal", "DLS")

    def test_parse_cached_value_no_prefix_treated_as_event(self, storenames_config_instance):
        result = storenames_config_instance._parse_cached_value("RewardedPort")
        assert result == ("event TTLs", "RewardedPort")

    def test_parse_cached_value_empty_string(self, storenames_config_instance):
        result = storenames_config_instance._parse_cached_value("")
        assert result == ("", "")

    # ── _get_help_text ────────────────────────────────────────────────────────

    def test_get_help_text_control(self, storenames_config_instance):
        assert storenames_config_instance._get_help_text("control") == "*Type appropriate region name*"

    def test_get_help_text_signal(self, storenames_config_instance):
        assert storenames_config_instance._get_help_text("signal") == "*Type appropriate region name*"

    def test_get_help_text_event_ttls(self, storenames_config_instance):
        assert storenames_config_instance._get_help_text("event TTLs") == "*Type event name for the TTLs*"

    def test_get_help_text_empty(self, storenames_config_instance):
        assert storenames_config_instance._get_help_text("") == ""

    # ── Cache pre-population ──────────────────────────────────────────────────

    def test_cache_control_pre_populates_dropdown_and_textbox(self, panel_extension, show_config_button):
        dropdowns = {}
        textboxes = {}
        StorenamesConfig(
            show_config_button=show_config_button,
            storename_dropdowns=dropdowns,
            storename_textboxes=textboxes,
            storenames=["Dv1A"],
            storenames_cache={"Dv1A": ["control_DMS"]},
        )
        assert dropdowns["Dv1A_0"].value == "control"
        assert textboxes["Dv1A_0"].value == "DMS"

    def test_cache_signal_pre_populates_dropdown_and_textbox(self, panel_extension, show_config_button):
        dropdowns = {}
        textboxes = {}
        StorenamesConfig(
            show_config_button=show_config_button,
            storename_dropdowns=dropdowns,
            storename_textboxes=textboxes,
            storenames=["Dv2A"],
            storenames_cache={"Dv2A": ["signal_DLS"]},
        )
        assert dropdowns["Dv2A_0"].value == "signal"
        assert textboxes["Dv2A_0"].value == "DLS"

    def test_cache_event_ttl_pre_populates_dropdown_and_textbox(self, panel_extension, show_config_button):
        dropdowns = {}
        textboxes = {}
        StorenamesConfig(
            show_config_button=show_config_button,
            storename_dropdowns=dropdowns,
            storename_textboxes=textboxes,
            storenames=["PrtR"],
            storenames_cache={"PrtR": ["RewardedPort"]},
        )
        assert dropdowns["PrtR_0"].value == "event TTLs"
        assert textboxes["PrtR_0"].value == "RewardedPort"

    # ── Dropdown change triggers help pane update ─────────────────────────────

    def test_dropdown_change_updates_help_pane(self, panel_extension, show_config_button):
        dropdowns = {}
        config = StorenamesConfig(
            show_config_button=show_config_button,
            storename_dropdowns=dropdowns,
            storename_textboxes={},
            storenames=["Dv1A"],
            storenames_cache={},
        )
        dropdown = dropdowns["Dv1A_0"]
        help_pane = config._dropdown_help_map[dropdown]

        dropdown.value = "control"

        assert help_pane.object == "*Type appropriate region name*"

    def test_dropdown_change_to_event_updates_help_pane(self, panel_extension, show_config_button):
        dropdowns = {}
        config = StorenamesConfig(
            show_config_button=show_config_button,
            storename_dropdowns=dropdowns,
            storename_textboxes={},
            storenames=["PrtR"],
            storenames_cache={},
        )
        dropdown = dropdowns["PrtR_0"]
        help_pane = config._dropdown_help_map[dropdown]

        dropdown.value = "event TTLs"

        assert help_pane.object == "*Type event name for the TTLs*"
