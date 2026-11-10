"""The shared pieces of GuPPy's editable tables: a column heading and a row's delete button."""

import panel as pn

from .typography import SECONDARY_INK, SEMIBOLD, SMALL_SIZE


def column_heading(*, text: str, width: int | None = None, margin: tuple[int, ...] = (5, 10)) -> pn.pane.HTML:
    """Return the heading over one column of an editable table.

    Parameters
    ----------
    text : str
        The column's name, in Title Case with any unit in parentheses.
    width : int, optional
        Width of the column, in pixels, so the heading lines up with the fields under it.
    margin : tuple of int, optional
        Margin around the heading, in Panel's order.

    Returns
    -------
    panel.pane.HTML
        The heading, set small, semibold and gray.
    """
    style = f"font-size: {SMALL_SIZE}; font-weight: {SEMIBOLD}; color: {SECONDARY_INK};"
    return pn.pane.HTML(f"<div style='{style}'>{text}</div>", width=width, margin=margin)


def delete_row_button(*, description: str) -> pn.widgets.Button:
    """Return the trash button that deletes one row of an editable table.

    Parameters
    ----------
    description : str
        Tooltip naming what the button deletes, such as "Delete this period".

    Returns
    -------
    panel.widgets.Button
        A light icon button the height of a text field.
    """
    return pn.widgets.Button(icon="trash", button_type="light", width=40, height=38, description=description)
