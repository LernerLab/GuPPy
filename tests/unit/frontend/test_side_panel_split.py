import panel as pn
import pytest

from guppy.frontend.side_panel_split import SidePanelSplit


@pytest.fixture
def split(panel_extension):
    return SidePanelSplit(
        main=pn.pane.Markdown("plot"), side=pn.widgets.MultiSelect(), side_label="Events", height=300, side_width=400
    )


class TestSidePanelSplit:
    def test_lays_out_plot_handle_and_side_panel_in_one_row(self, split):
        row = split.__panel__()

        assert isinstance(row, pn.Row)
        assert row.sizing_mode == "stretch_width"
        assert list(row) == [row[0], split.handle, split.side]
        assert split.side.width == 400
        assert split.handle.height == 300

    def test_dragging_the_handle_resizes_the_side_panel(self, split):
        split.handle.side_width = 520

        assert split.side.width == 520

    def test_the_handle_button_hides_and_shows_the_side_panel(self, split):
        split.handle.collapsed = True
        assert split.side.visible is False

        split.handle.collapsed = False
        assert split.side.visible is True

    def test_the_button_tooltip_names_what_the_panel_holds(self, split):
        assert split.handle.side_label == "Events"
