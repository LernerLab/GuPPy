"""Shared pieces of the Step 6 NWB metadata form: field labels, the required mark, and its text chrome.

The form uses the homepage's cards and Panel's default widgets. Custom text is rendered as
inline-styled HTML in Markdown panes, since inline styles reach inside Panel's shadow roots.
"""

import panel as pn

from .typography import (
    BODY_SIZE,
    INK,
    MONOSPACE_FAMILY,
    REGULAR,
    SECONDARY_INK,
    SEMIBOLD,
    SMALL_SIZE,
    title_case,
)
from ..visualization.palette import CONTROL, SIGNAL

# Bootstrap's danger red, which the rest of the app's alerts use.
REQUIRED_RED = "#DC3545"
RULE = "#DEE2E6"

# Appended to a required field: a red asterisk after its label.
REQUIRED_STYLESHEET = f"""
:host label::after {{
  content: " *";
  color: {REQUIRED_RED};
  font-weight: {SEMIBOLD};
}}
"""

# ----------------------------------------------------------------------------------------------------------------------
# Label humanizing
# ----------------------------------------------------------------------------------------------------------------------
_ACRONYMS = {
    "na": "NA",
    "ap": "AP",
    "ml": "ML",
    "dv": "DV",
    "id": "ID",
    "rrid": "RRID",
    "doi": "DOI",
}

# Word pairs in a key that a label writes as one hyphenated word.
_COMPOUNDS = {"cut_on": "cut-on", "cut_off": "cut-off"}

# How each word after "_in_" in a key is written in the label's unit.
_UNIT_SYMBOLS = {
    "nm": "nm",
    "um": "µm",
    "mm": "mm",
    "nl": "nL",
    "ul": "µL",
    "ml": "mL",
    "w": "W",
    "s": "s",
    "deg": "deg",
    "degrees": "deg",
    "m2": "m²",
    "vg": "vg",
    "per": "/",
}


def humanize(name: str) -> str:
    """Turn a snake_case metadata key into a Title Case field label, with its unit in parentheses.

    A key such as ``volume_in_uL`` or ``intensity_in_W_per_m2`` reads "Volume (µL)" or "Intensity (W/m²)".
    """
    for compound, hyphenated in _COMPOUNDS.items():
        name = name.replace(compound, hyphenated)
    words = name.split("_")
    unit = ""
    if "in" in words[1:]:
        index = words.index("in", 1)
        unit = " (" + "".join(_UNIT_SYMBOLS[word.lower()] for word in words[index + 1 :]) + ")"
        words = words[:index]
    return title_case(" ".join(_ACRONYMS.get(word, word) for word in words)) + unit


# ----------------------------------------------------------------------------------------------------------------------
# Inline-styled HTML chrome (Markdown panes; inline styles bypass the shadow boundary)
# ----------------------------------------------------------------------------------------------------------------------
def _markdown(html: str, **styles: str) -> pn.pane.Markdown:
    # Panel's default horizontal margin, so text lines up with the fields beside and below it.
    return pn.pane.Markdown(html, sizing_mode="stretch_width", styles=styles, margin=(0, 10))


def section_label(text: str, note: str = "") -> pn.pane.Markdown:
    """A section label with an optional trailing note, over a hairline rule."""
    note_html = (
        f"<span style='font-weight:{REGULAR};color:{SECONDARY_INK};font-size:{SMALL_SIZE};margin-left:10px;'>{note}</span>"
        if note
        else ""
    )
    html = (
        f"<div style='display:flex;align-items:baseline;border-bottom:1px solid {RULE};padding-bottom:7px;margin:6px 0 2px 0;'>"
        f"<span style='font-size:{BODY_SIZE};font-weight:{SEMIBOLD};color:{INK};'>{text}</span>"
        f"{note_html}</div>"
    )
    return _markdown(html)


def subgroup_label(text: str) -> pn.pane.Markdown:
    """A quieter label that groups several fields (e.g. a fiber's insertion)."""
    html = (
        f"<div style='font-size:{BODY_SIZE};font-weight:{SEMIBOLD};color:{INK};margin:14px 2px 2px 2px;'>{text}</div>"
    )
    return _markdown(html)


def help_note(text: str) -> pn.pane.Markdown:
    """Muted helper text under a heading."""
    return _markdown(f"<div style='font-size:{SMALL_SIZE};color:{SECONDARY_INK};line-height:1.45;'>{text}</div>")


def intro_note(html: str) -> pn.pane.Markdown:
    """A short page-top instruction paragraph (slightly more present than help_note)."""
    return _markdown(
        f"<div style='font-size:{BODY_SIZE};color:{INK};line-height:1.45;max-width:780px;margin:2px 0 4px 0;'>{html}</div>"
    )


def channel_role_chip(role: str, store_name: str) -> pn.pane.Markdown:
    """A header for one channel: its role, in the role's plot color, beside the store it comes from."""
    color = SIGNAL if role == "signal" else CONTROL
    html = (
        f"<div style='display:flex;align-items:center;gap:10px;'>"
        f"<span style='font-size:{SMALL_SIZE};font-weight:{SEMIBOLD};color:{color};background:{color}1A;"
        f"padding:2px 11px;border-radius:20px;'>{title_case(role)}</span>"
        f"<span style='font-size:{SMALL_SIZE};color:{SECONDARY_INK};font-family:{MONOSPACE_FAMILY};'>{store_name}</span>"
        f"</div>"
    )
    return _markdown(html)
