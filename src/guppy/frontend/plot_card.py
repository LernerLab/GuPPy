"""How a Visualization plot is laid out: a short note, one toolbar of labeled controls, then the plot.

On a tab with several plots, each sits in a homepage card, which can be collapsed to focus on another.
On a tab with one, the same layout sits on the page with no card around it.
"""

import panel as pn

from .input_parameters import CARD_STYLES, SECTION_STYLES
from .side_panel_split import SidePanelSplit
from .typography import SECONDARY_INK, SEMIBOLD, SMALL_CLASS, SMALL_SIZE

# Each control's own margin inside a toolbar field, so neighbouring controls share one gap.
CONTROL_MARGIN = (0, 5)

# Wide enough for a session and run path, which is what tells runs from different sessions apart.
RUN_FOLDER_WIDTH = 560

# A checkbox is shorter than a dropdown, so it drops to sit level with the dropdowns beside it.
CHECKBOX_MARGIN = (8, 5, 0, 5)

_RULE_COLOR = "#E9ECEF"
_LABEL_STYLE = f"font-size: {SMALL_SIZE}; font-weight: {SEMIBOLD}; color: {SECONDARY_INK}; white-space: nowrap;"


def toolbar_field(label: str, *controls: pn.viewable.Viewable) -> pn.Column:
    """One labeled group in a toolbar: a small gray label over the controls it names.

    Parameters
    ----------
    label : str
        Text shown above the controls.
    *controls : pn.viewable.Viewable
        The controls, side by side, in reading order.

    Returns
    -------
    panel.Column
        The label above its controls.
    """
    heading = pn.pane.HTML(f"<div style='{_LABEL_STYLE}'>{label}</div>", margin=(0, 5, 4, 5))
    return pn.Column(heading, pn.Row(*controls, margin=0), margin=(0, 8))


def range_separator() -> pn.pane.HTML:
    """The dash between the low and high boxes of a range."""
    return pn.pane.HTML(f"<span style='color: {SECONDARY_INK};'>–</span>", margin=(0, 2), align="center")


def side_column(label: str, picker: pn.viewable.Viewable) -> pn.Column:
    """A labeled picker, stretched to the width of the side panel it sits in.

    Parameters
    ----------
    label : str
        Text shown above the picker.
    picker : pn.viewable.Viewable
        The picker choosing what the plot shows.

    Returns
    -------
    panel.Column
        The label above the picker.
    """
    heading = pn.pane.HTML(f"<div style='{_LABEL_STYLE}'>{label}</div>", margin=(10, 5, 4, 5))
    return pn.Column(heading, picker, sizing_mode="stretch_width", margin=(8, 10, 10, 0))


def plot_with_side_panel(
    plot: pn.viewable.Viewable, *, label: str, picker: pn.viewable.Viewable, height: object, side_width: int = 460
) -> SidePanelSplit:
    """A plot with its picker in a side panel that can be resized or hidden.

    Parameters
    ----------
    plot : pn.viewable.Viewable
        The plot, which takes the width the side panel leaves.
    label : str
        What the picker chooses (e.g. "Events"), shown above it.
    picker : pn.viewable.Viewable
        The picker, stretched to the side panel's width.
    height : int or bound function
        The plot's height in pixels, which the handle between the two matches.
    side_width : int, optional
        The side panel's starting width in pixels.

    Returns
    -------
    SidePanelSplit
        The plot and the side panel, stretched to the width of their container.
    """
    return SidePanelSplit(
        main=plot, side=side_column(label, picker), side_label=label, height=height, side_width=side_width
    )


def _plot_layout(
    *,
    note: str,
    fields: list[pn.viewable.Viewable],
    trailing_field: pn.viewable.Viewable | None,
    body: list[pn.viewable.Viewable],
) -> list[pn.viewable.Viewable]:
    """The note (when there is one), the toolbar and the body, with a thin rule under the toolbar when a body follows."""
    note_panes = (
        [pn.pane.Markdown(note, sizing_mode="stretch_width", css_classes=[SMALL_CLASS], margin=(5, 10, 0, 10))]
        if note
        else []
    )
    trailing = []
    if trailing_field is not None:
        # An auto left margin holds the field at the right end of whichever line it lands on.
        trailing_field.styles = {**trailing_field.styles, "margin-left": "auto"}
        trailing = [trailing_field]
    toolbar_styles = {"border-bottom": f"1px solid {_RULE_COLOR}", "padding-bottom": "12px"} if body else {}
    # Fields wrap onto another line when the window is too narrow for one, never past its edge.
    toolbar = pn.FlexBox(
        *fields,
        *trailing,
        flex_wrap="wrap",
        align_items="flex-start",
        sizing_mode="stretch_width",
        styles=toolbar_styles,
        margin=(4, 10, 8, 10),
    )
    return [*note_panes, toolbar, *body]


def plot_card(
    *,
    title: str,
    note: str,
    fields: list[pn.viewable.Viewable],
    trailing_field: pn.viewable.Viewable | None = None,
    body: list[pn.viewable.Viewable] | None = None,
) -> pn.Card:
    """A homepage card holding one white section: a note, one toolbar row, then the plot.

    Parameters
    ----------
    title : str
        The card's title.
    note : str
        Small gray text above the toolbar, saying what the card shows.
    fields : list of pn.viewable.Viewable
        Toolbar fields from the left, in reading order.
    trailing_field : pn.viewable.Viewable, optional
        A toolbar field held at the right edge, such as the save controls.
    body : list of pn.viewable.Viewable, optional
        What sits under the toolbar, usually the plot. A thin rule separates the two.

    Returns
    -------
    panel.Card
        The card, stretched to the width of its container.
    """
    section = pn.WidgetBox(
        *_plot_layout(note=note, fields=fields, trailing_field=trailing_field, body=body or []),
        sizing_mode="stretch_width",
        styles=SECTION_STYLES,
    )
    return pn.Card(section, title=title, styles=dict(CARD_STYLES), sizing_mode="stretch_width")


def plot_page(
    *,
    fields: list[pn.viewable.Viewable],
    trailing_field: pn.viewable.Viewable | None = None,
    body: list[pn.viewable.Viewable] | None = None,
    note: str = "",
) -> pn.Column:
    """The layout of a plot card with no card around it, for a tab that shows only one plot.

    Parameters
    ----------
    fields : list of pn.viewable.Viewable
        Toolbar fields from the left, in reading order.
    trailing_field : pn.viewable.Viewable, optional
        A toolbar field held at the right edge, such as the save controls.
    body : list of pn.viewable.Viewable, optional
        What sits under the toolbar, usually the plot. A thin rule separates the two.
    note : str, optional
        Small gray text above the toolbar, for what the page's controls and plot do not say themselves.

    Returns
    -------
    panel.Column
        The note, toolbar and body, stretched to the width of the tab.
    """
    return pn.Column(
        *_plot_layout(note=note, fields=fields, trailing_field=trailing_field, body=body or []),
        sizing_mode="stretch_width",
    )
