"""The card every Visualization plot sits in: a titled header, one toolbar of labeled controls, then the plot."""

import panel as pn

from .typography import HEADING_SIZE, INK, SECONDARY_INK, SEMIBOLD, SMALL_SIZE

# Each control's own margin inside a toolbar field, so neighbouring controls share one gap.
CONTROL_MARGIN = (0, 5)

# A checkbox is shorter than a dropdown, so it drops to sit level with the dropdowns beside it.
CHECKBOX_MARGIN = (8, 5, 0, 5)

_CARD_STYLES = {"background": "White", "border": "1px solid #C8CCD0", "border-radius": "6px"}
_HEADER_BACKGROUND = "#F8F9FA"
_FLAT_STYLESHEET = ":host { box-shadow: none !important; }"
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
    """A labeled picker beside a plot, set off from it by a thin rule.

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
    return pn.Column(
        heading,
        picker,
        styles={"border-left": f"1px solid {_RULE_COLOR}", "padding-left": "8px"},
        margin=(8, 10, 10, 0),
    )


def plot_card(
    *,
    title: str,
    note: str,
    fields: list[pn.viewable.Viewable],
    trailing_field: pn.viewable.Viewable | None = None,
    body: list[pn.viewable.Viewable] | None = None,
) -> pn.Card:
    """A card headed by its title and note, with one toolbar row over the plot.

    Parameters
    ----------
    title : str
        The card's title.
    note : str
        Small gray text beside the title, saying what the card shows.
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
    body = body or []
    header = pn.pane.HTML(
        f"<span style='font-size: {HEADING_SIZE}; font-weight: {SEMIBOLD}; color: {INK};'>{title}</span>"
        f"<span style='font-size: {SMALL_SIZE}; color: {SECONDARY_INK}; margin-left: 12px;'>{note}</span>",
        margin=(0, 5),
    )
    trailing = [pn.Spacer(sizing_mode="stretch_width"), trailing_field] if trailing_field is not None else []
    toolbar_styles = {"border-bottom": f"1px solid {_RULE_COLOR}", "padding-bottom": "12px"} if body else {}
    toolbar = pn.Row(*fields, *trailing, sizing_mode="stretch_width", styles=toolbar_styles, margin=(4, 10, 8, 10))
    return pn.Card(
        toolbar,
        *body,
        header=header,
        collapsible=False,
        sizing_mode="stretch_width",
        header_background=_HEADER_BACKGROUND,
        styles=_CARD_STYLES,
        stylesheets=[_FLAT_STYLESHEET],
        margin=(5, 10, 16, 10),
    )
