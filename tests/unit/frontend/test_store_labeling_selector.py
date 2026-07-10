from guppy.frontend.store_labeling_selector import StoreLabelingSelector


class TestStoreLabelingSelector:
    def test_empty_allnames_shows_alert(self, panel_extension):
        selector = StoreLabelingSelector(allnames=[])
        assert "No store_ids found" in selector.alert.object
        assert selector.cross_selector.options == []

    def test_populated_allnames_sets_options(self, panel_extension):
        allnames = ["Dv1A", "Dv2A", "PrtR"]
        selector = StoreLabelingSelector(allnames=allnames)
        assert selector.cross_selector.options == allnames
        assert selector.multi_choice.options == allnames
        assert selector.cross_selector.value == []

    def test_repeat_stores_true_populates_widget_box(self, panel_extension):
        selector = StoreLabelingSelector(allnames=["Dv1A"])
        selector.repeat_stores.value = True
        assert len(selector.repeat_store_wd.objects) > 0
        assert selector.multi_choice in selector.repeat_store_wd.objects
        assert selector.literal_input_1 in selector.repeat_store_wd.objects

    def test_repeat_stores_false_clears_widget_box(self, panel_extension):
        selector = StoreLabelingSelector(allnames=["Dv1A"])
        selector.repeat_stores.value = True
        selector.repeat_stores.value = False
        assert selector.repeat_store_wd.objects == []

    def test_get_cross_selector_returns_value(self, panel_extension):
        selector = StoreLabelingSelector(allnames=["Dv1A", "Dv2A"])
        assert selector.get_cross_selector() == selector.cross_selector.value

    def test_set_select_location_options(self, panel_extension):
        selector = StoreLabelingSelector(allnames=["Dv1A"])
        selector.set_select_location_options(["option_a", "option_b"])
        assert selector.select_location.options == ["option_a", "option_b"]

    def test_set_alert_message(self, panel_extension):
        selector = StoreLabelingSelector(allnames=["Dv1A"])
        selector.set_alert_message("test message")
        assert selector.alert.object == "test message"

    def test_set_and_get_literal_input_2_round_trips(self, panel_extension):
        selector = StoreLabelingSelector(allnames=["Dv1A"])
        selector.set_literal_input_2({"key": "val"})
        result = selector.get_literal_input_2()
        assert result == {"key": "val"}

    def test_set_path(self, panel_extension):
        selector = StoreLabelingSelector(allnames=["Dv1A"])
        selector.set_path("/some/path")
        assert selector.path.value == "/some/path"

    def test_get_take_widgets_returns_values(self, panel_extension):
        selector = StoreLabelingSelector(allnames=["Dv1A", "Dv2A"])
        result = selector.get_take_widgets()
        assert result == [selector.multi_choice.value, selector.literal_input_1.value]

    def test_set_change_widgets(self, panel_extension):
        selector = StoreLabelingSelector(allnames=["Dv1A"])
        selector.set_change_widgets(["Dv1A"])
        assert selector.text.value == ["Dv1A"]

    def test_configure_store_ids_visible_when_store_ids_present(self, panel_extension):
        selector = StoreLabelingSelector(allnames=["Dv1A"])
        selector.store_ids = ["Dv1A"]
        selector.configure_store_ids(store_id_to_store_labels={})
        assert selector.store_id_config_widgets.visible is True

    def test_configure_store_ids_hidden_when_store_ids_empty(self, panel_extension):
        selector = StoreLabelingSelector(allnames=["Dv1A"])
        selector.store_ids = []
        selector.configure_store_ids(store_id_to_store_labels={})
        assert selector.store_id_config_widgets.visible is False
