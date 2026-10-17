import types

import panel as pn
import pytest

from guppy.frontend.store_labeling_selector import StoreLabelingSelector


class FakeNotification:
    def __init__(self, message, duration):
        self.message = message
        self.duration = duration
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


class FakeNotificationArea:
    """Stands in for the browser session's notification area, which headless tests do not have."""

    def __init__(self):
        self.shown = []

    def success(self, message, *, duration):
        notification = FakeNotification(message, duration)
        self.shown.append(notification)
        return notification


@pytest.fixture
def selector(panel_extension):
    return StoreLabelingSelector(allnames=["Dv1A"])


@pytest.fixture
def notification_area(selector, monkeypatch):
    area = FakeNotificationArea()
    monkeypatch.setattr(pn, "state", types.SimpleNamespace(notifications=area))
    return area


class TestStoreLabelingSelector:
    def test_empty_allnames_shows_alert(self, panel_extension):
        selector = StoreLabelingSelector(allnames=[])
        assert "No store_ids found" in selector.alert.object
        assert selector.alert.visible is True
        assert selector.cross_selector.options == []

    def test_alert_hidden_when_there_is_nothing_to_report(self, selector):
        assert selector.alert.visible is False

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

    def test_set_events_replaces_options_in_both_widgets(self, panel_extension):
        selector = StoreLabelingSelector(allnames=[])
        selector.set_events(events=["bl72bl82_12feb2024_fp_415nm_G0", "bl72bl82_12feb2024_fp_470nm_G0", "event0"])
        assert selector.cross_selector.options == [
            "bl72bl82_12feb2024_fp_415nm_G0",
            "bl72bl82_12feb2024_fp_470nm_G0",
            "event0",
        ]
        assert selector.multi_choice.options == [
            "bl72bl82_12feb2024_fp_415nm_G0",
            "bl72bl82_12feb2024_fp_470nm_G0",
            "event0",
        ]

    def test_set_select_location_options_labels_runs_by_folder_name(self, selector):
        selector.set_select_location_options(["/data/session/session_output_1", "/data/session/session_output_2"])
        assert selector.select_location.options == {
            "session_output_1": "/data/session/session_output_1",
            "session_output_2": "/data/session/session_output_2",
        }
        assert selector.select_location.value == "/data/session/session_output_1"

    def test_set_select_location_options_keeps_a_value_still_offered(self, selector):
        selector.set_select_location_options(["/data/session/session_output_1", "/data/session/session_output_2"])
        selector.select_location.value = "/data/session/session_output_2"
        selector.set_select_location_options(["/data/session/session_output_2", "/data/session/session_output_3"])
        assert selector.select_location.value == "/data/session/session_output_2"

    def test_set_alert_message(self, selector):
        selector.set_alert_message("test message")
        assert selector.alert.object == "test message"
        assert selector.alert.visible is True

    def test_set_alert_message_no_alerts_hides_the_pane(self, selector):
        selector.set_alert_message("test message")
        selector.set_alert_message("#### No alerts !!")
        assert selector.alert.visible is False

    def test_opens_in_create_mode_with_run_picker_hidden(self, selector):
        assert selector.get_overwrite_mode() == "create_new_file"
        assert selector.run_name.visible is True
        assert selector.select_location.visible is False

    def test_overwrite_mode_watcher_swaps_run_name_for_run_picker(self, selector):
        modes = []
        selector.attach_overwrite_mode_watcher(lambda event: modes.append(event.new))
        selector.overwrite_mode.value = "over_write_file"
        assert modes == ["over_write_file"]
        assert selector.get_overwrite_mode() == "over_write_file"
        assert selector.run_name.visible is False
        assert selector.select_location.visible is True

    def test_overwrite_mode_watcher_restores_run_name_when_creating_again(self, selector):
        modes = []
        selector.attach_overwrite_mode_watcher(lambda event: modes.append(event.new))
        selector.overwrite_mode.value = "over_write_file"
        selector.overwrite_mode.value = "create_new_file"
        assert modes == ["over_write_file", "create_new_file"]
        assert selector.run_name.visible is True
        assert selector.select_location.visible is False

    def test_set_run_name_fires_the_run_name_watcher(self, selector):
        run_names = []
        selector.attach_run_name_watcher(lambda event: run_names.append(event.new))
        selector.set_run_name("filter_100")
        assert selector.get_run_name() == "filter_100"
        assert run_names == ["filter_100"]

    def test_set_and_get_literal_input_2_round_trips(self, panel_extension):
        selector = StoreLabelingSelector(allnames=["Dv1A"])
        selector.set_literal_input_2({"key": "val"})
        result = selector.get_literal_input_2()
        assert result == {"key": "val"}

    def test_show_saved_message_pops_up_until_dismissed(self, selector, notification_area):
        selector.show_saved_message("Saved.")
        assert [(notification.message, notification.duration) for notification in notification_area.shown] == [
            ("Saved.", 0)
        ]

    def test_show_saved_message_replaces_the_earlier_one(self, selector, notification_area):
        selector.show_saved_message("First save.")
        selector.show_saved_message("Second save.")
        assert [notification.destroyed for notification in notification_area.shown] == [True, False]

    def test_hide_saved_message_dismisses_it(self, selector, notification_area):
        selector.show_saved_message("Saved.")
        selector.hide_saved_message()
        assert notification_area.shown[0].destroyed is True

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
