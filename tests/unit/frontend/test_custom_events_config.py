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

    def test_alert_is_hidden_until_there_is_a_message(self, config):
        assert config.alert.visible is False

        config.set_alert_message("#### hello")

        assert config.alert.object == "#### hello"
        assert config.alert.visible is True

    def test_the_delete_button_removes_its_row(self, config):
        config.add_event_row()
        config.rows[0][0].value = "movement_onset"
        config.rows[1][0].value = "lever_press"
        first_delete = config.event_rows[0][2]

        first_delete.clicks += 1

        assert [name.value for name, _ in config.rows] == ["lever_press"]
        assert len(config.event_rows) == 1

    def test_rows_sit_under_column_headings(self, config):
        heading = config.widget[1]
        assert [pane.object.split(">")[1].split("<")[0] for pane in heading] == ["Event Name", "Timestamps (s)"]

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
