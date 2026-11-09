import panel as pn

from guppy.frontend.plot_card import (
    plot_card,
    plot_with_side_panel,
    side_column,
    toolbar_field,
)
from guppy.frontend.side_panel_split import SidePanelSplit


def label_text(pane):
    return pane.object.split(">")[1].split("<")[0]


class TestToolbarField:
    def test_labels_its_controls(self, panel_extension):
        first, second = pn.widgets.Select(), pn.widgets.Button()

        field = toolbar_field("Save Plot", first, second)

        assert label_text(field[0]) == "Save Plot"
        assert list(field[1]) == [first, second]


class TestSideColumn:
    def test_labels_its_picker(self, panel_extension):
        picker = pn.widgets.MultiSelect()

        column = side_column("Trials", picker)

        assert label_text(column[0]) == "Trials"
        assert column[1] is picker


class TestPlotWithSidePanel:
    def test_puts_the_labeled_picker_beside_the_plot(self, panel_extension):
        plot, picker = pn.pane.Markdown("plot"), pn.widgets.MultiSelect()

        split = plot_with_side_panel(plot, label="Trials", picker=picker, height=300, side_width=320)

        assert isinstance(split, SidePanelSplit)
        main, handle, side = split.__panel__()
        assert main is plot
        assert (handle.side_label, handle.height, side.width) == ("Trials", 300, 320)
        assert label_text(side[0][0]) == "Trials"
        assert side[0][1] is picker


class TestPlotCard:
    def test_header_carries_title_and_note(self, panel_extension):
        card = plot_card(title="Selected Trials", note="Trials of one event.", fields=[])

        assert ">Selected Trials</span>" in card.header.object
        assert ">Trials of one event.</span>" in card.header.object
        assert card.collapsible is True

    def test_toolbar_comes_first_then_the_body(self, panel_extension):
        field = toolbar_field("Event", pn.widgets.Select())
        plot = pn.pane.Markdown("plot")

        card = plot_card(title="Title", note="Note.", fields=[field], body=[plot])

        assert list(card) == [card[0], plot]
        assert list(card[0]) == [field]

    def test_trailing_field_is_pushed_to_the_right_edge(self, panel_extension):
        leading = toolbar_field("Event", pn.widgets.Select())
        trailing = toolbar_field("Save Plot", pn.widgets.Button())

        card = plot_card(title="Title", note="Note.", fields=[leading], trailing_field=trailing)

        toolbar = card[0]
        assert toolbar[0] is leading
        assert isinstance(toolbar[1], pn.Spacer) and toolbar[1].sizing_mode == "stretch_width"
        assert toolbar[2] is trailing

    def test_rule_under_the_toolbar_only_when_a_body_follows(self, panel_extension):
        settings_only = plot_card(title="Title", note="Note.", fields=[])
        with_plot = plot_card(title="Title", note="Note.", fields=[], body=[pn.pane.Markdown("plot")])

        assert "border-bottom" not in settings_only[0].styles
        assert with_plot[0].styles["border-bottom"] == "1px solid #E9ECEF"
