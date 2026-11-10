import panel as pn

from guppy.frontend.row_controls import column_heading, delete_row_button


class TestColumnHeading:
    def test_is_small_semibold_gray_text(self):
        heading = column_heading(text="Start (s)", width=110)

        assert heading.object == "<div style='font-size: 12px; font-weight: 600; color: #6C757D;'>Start (s)</div>"
        assert heading.width == 110
        assert heading.margin == (5, 10)


class TestDeleteRowButton:
    def test_is_a_light_trash_icon_named_by_its_tooltip(self):
        button = delete_row_button(description="Delete this period")

        assert isinstance(button, pn.widgets.Button)
        assert button.icon == "trash"
        assert button.button_type == "light"
        assert (button.width, button.height) == (40, 38)
        assert button.description == "Delete this period"
