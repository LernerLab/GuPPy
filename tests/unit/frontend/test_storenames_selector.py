from guppy.frontend.storenames_selector import StorenamesSelector


def test_storenames_selector_empty_allnames_shows_alert(panel_extension):
    selector = StorenamesSelector(allnames=[])
    assert "No storenames found" in selector.alert.object
    assert selector.cross_selector.options == []


def test_storenames_selector_populated_allnames_sets_options(panel_extension):
    allnames = ["Dv1A", "Dv2A", "PrtR"]
    selector = StorenamesSelector(allnames=allnames)
    assert selector.cross_selector.options == allnames
    assert selector.multi_choice.options == allnames
    assert selector.cross_selector.value == []


def test_storenames_selector_repeat_storenames_true_populates_widget_box(panel_extension):
    selector = StorenamesSelector(allnames=["Dv1A"])
    selector.repeat_storenames.value = True
    assert len(selector.repeat_storename_wd.objects) > 0
    assert selector.multi_choice in selector.repeat_storename_wd.objects
    assert selector.literal_input_1 in selector.repeat_storename_wd.objects


def test_storenames_selector_repeat_storenames_false_clears_widget_box(panel_extension):
    selector = StorenamesSelector(allnames=["Dv1A"])
    selector.repeat_storenames.value = True
    selector.repeat_storenames.value = False
    assert selector.repeat_storename_wd.objects == []


def test_storenames_selector_get_cross_selector_returns_value(panel_extension):
    selector = StorenamesSelector(allnames=["Dv1A", "Dv2A"])
    assert selector.get_cross_selector() == selector.cross_selector.value


def test_storenames_selector_set_select_location_options(panel_extension):
    selector = StorenamesSelector(allnames=["Dv1A"])
    selector.set_select_location_options(["option_a", "option_b"])
    assert selector.select_location.options == ["option_a", "option_b"]


def test_storenames_selector_set_alert_message(panel_extension):
    selector = StorenamesSelector(allnames=["Dv1A"])
    selector.set_alert_message("test message")
    assert selector.alert.object == "test message"


def test_storenames_selector_set_and_get_literal_input_2_round_trips(panel_extension):
    selector = StorenamesSelector(allnames=["Dv1A"])
    selector.set_literal_input_2({"key": "val"})
    result = selector.get_literal_input_2()
    assert result == {"key": "val"}


def test_storenames_selector_set_path(panel_extension):
    selector = StorenamesSelector(allnames=["Dv1A"])
    selector.set_path("/some/path")
    assert selector.path.value == "/some/path"


def test_storenames_selector_get_take_widgets_returns_values(panel_extension):
    selector = StorenamesSelector(allnames=["Dv1A", "Dv2A"])
    result = selector.get_take_widgets()
    assert result == [selector.multi_choice.value, selector.literal_input_1.value]


def test_storenames_selector_set_change_widgets(panel_extension):
    selector = StorenamesSelector(allnames=["Dv1A"])
    selector.set_change_widgets(["Dv1A"])
    assert selector.text.value == ["Dv1A"]


def test_storenames_selector_configure_storenames_visible_when_storenames_present(panel_extension):
    import panel as pn

    show_config_button = pn.widgets.Button(name="Show")
    storename_dropdowns = {}
    storename_textboxes = {}
    selector = StorenamesSelector(allnames=["Dv1A"])
    selector.configure_storenames(
        storename_dropdowns=storename_dropdowns,
        storename_textboxes=storename_textboxes,
        storenames=["Dv1A"],
        storenames_cache={},
    )
    assert selector.storename_config_widgets.visible is True


def test_storenames_selector_configure_storenames_hidden_when_storenames_empty(panel_extension):

    selector = StorenamesSelector(allnames=["Dv1A"])
    selector.configure_storenames(
        storename_dropdowns={},
        storename_textboxes={},
        storenames=[],
        storenames_cache={},
    )
    assert selector.storename_config_widgets.visible is False
