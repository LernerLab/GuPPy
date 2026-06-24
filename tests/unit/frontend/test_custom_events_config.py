import panel as pn
import pytest

from guppy.frontend.custom_events_config import CustomEventsConfig


@pytest.fixture
def config(panel_extension):
    return CustomEventsConfig()


class TestCustomEventsConfig:
    def test_starts_with_one_row(self, config):
        assert len(config.rows) == 1

    def test_add_event_row_grows_rows(self, config):
        config.add_event_row()
        config.add_event_row()
        assert len(config.rows) == 3

    def test_get_rows_returns_entered_pairs(self, config):
        config.rows[0][0].value = "movement_onset"
        config.rows[0][1].value = "0.1\n0.2"
        config.add_event_row()
        config.rows[1][0].value = "lever_press"
        config.rows[1][1].value = "3.0"
        assert config.get_rows() == [("movement_onset", "0.1\n0.2"), ("lever_press", "3.0")]

    def test_set_alert_message(self, config):
        config.set_alert_message("#### hello")
        assert config.alert.object == "#### hello"

    def test_attach_callbacks_binds_handlers(self, config):
        calls = []
        config.attach_callbacks(
            {"add_event": lambda event=None: calls.append("add"), "save": lambda event=None: calls.append("save")}
        )
        config.add_event.clicks += 1
        config.save.clicks += 1
        assert calls == ["add", "save"]

    def test_widget_is_a_panel_column(self, config):
        assert isinstance(config.widget, pn.Column)
