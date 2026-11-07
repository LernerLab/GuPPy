"""Design system for the Step 6 NWB metadata form ("Clean clinical" register).

Light theme, restrained palette: tinted-cool neutrals carrying one calm blue
accent. Panel renders every widget in its own shadow root, so internals can only
be reached through a per-widget ``stylesheets=[":host ..."]`` injection; custom
text is rendered as inline-styled HTML in Markdown panes (inline styles are not
shadow-blocked). This module centralizes the tokens, the per-widget stylesheets,
and small helpers that build the styled chrome, so the selector stays readable.
"""

import panel as pn

from .typography import (
    BODY_SIZE,
    HEADING_SIZE,
    MONOSPACE_FAMILY,
    REGULAR,
    SECONDARY_INK,
    SEMIBOLD,
    SMALL_SIZE,
    title_case,
)
from .typography import INK as TYPE_INK

# ----------------------------------------------------------------------------------------------------------------------
# Tokens (cool, lightly tinted neutrals + one blue accent; no pure black/white)
# ----------------------------------------------------------------------------------------------------------------------
INK = TYPE_INK  # primary text
INK_SOFT = SECONDARY_INK  # secondary text
INK_FAINT = "#9AA7B4"  # placeholders / muted
ACCENT = "#2D6CDF"  # primary actions, focus, required mark
ACCENT_DARK = "#2156BC"  # hover/active
ACCENT_SOFT = "#EAF1FC"  # accent tint fills
PAGE = "#F4F6F9"  # page background (cool off-white)
SURFACE = "#FFFFFF"
SURFACE_INSET = "#F8FAFC"  # tinted strip / category fill
BORDER = "#E2E8F0"  # hairline
BORDER_STRONG = "#CBD5E1"
DANGER = "#D92D20"
SUCCESS = "#1E7B43"
HEADER_BG = "#1E2A36"  # template header: deep slate (calmer than bright blue)

_FOCUS_RING = "0 0 0 3px rgba(45,108,223,.16)"

# ----------------------------------------------------------------------------------------------------------------------
# Per-widget stylesheets (target the shadow-root internals via :host)
# ----------------------------------------------------------------------------------------------------------------------
INPUT_STYLESHEET = f"""
:host label {{
  color: {INK};
  font-size: {BODY_SIZE};
  font-weight: {REGULAR};
  margin-bottom: 5px;
}}
:host .bk-input {{
  border: 1px solid {BORDER_STRONG};
  border-radius: 8px;
  background: {SURFACE};
  color: {INK};
  padding: 8px 11px;
  box-shadow: none;
  transition: border-color .15s ease, box-shadow .15s ease;
}}
:host .bk-input::placeholder {{ color: {INK_FAINT}; }}
:host .bk-input:hover {{ border-color: #B7C2CE; }}
:host .bk-input:focus {{
  border-color: {ACCENT};
  box-shadow: {_FOCUS_RING};
  outline: none;
}}
:host select.bk-input {{ cursor: pointer; }}
:host .bk-spin-btn {{ border-color: {BORDER_STRONG}; color: {INK_SOFT}; }}
"""

# Appended after INPUT_STYLESHEET on required fields: a single accent-red asterisk.
REQUIRED_STYLESHEET = f"""
:host label::after {{
  content: " *";
  color: {DANGER};
  font-weight: {SEMIBOLD};
}}
"""

_BTN_BASE = f"""
:host .bk-btn {{
  border-radius: 8px;
  font-weight: {REGULAR};
  padding: 9px 17px;
  transition: background .15s ease, border-color .15s ease, box-shadow .15s ease, color .15s ease;
}}
"""

BUTTON_ACCENT = (
    _BTN_BASE
    + f"""
:host .bk-btn {{
  background: {ACCENT} !important;
  border: 1px solid {ACCENT} !important;
  color: #FFFFFF !important;
  box-shadow: 0 1px 2px rgba(16,24,40,.12);
}}
:host .bk-btn:hover {{ background: {ACCENT_DARK} !important; border-color: {ACCENT_DARK} !important; box-shadow: 0 3px 8px rgba(45,108,223,.28); }}
:host .bk-btn:active {{ background: #1B4699 !important; }}
"""
)

BUTTON_OUTLINE = (
    _BTN_BASE
    + f"""
:host .bk-btn {{
  background: {SURFACE} !important;
  border: 1px solid {BORDER_STRONG} !important;
  color: {ACCENT} !important;
}}
:host .bk-btn:hover {{ background: {ACCENT_SOFT} !important; border-color: {ACCENT} !important; }}
"""
)

BUTTON_GHOST = (
    _BTN_BASE
    + f"""
:host .bk-btn {{
  background: {SURFACE} !important;
  border: 1px solid {BORDER} !important;
  color: {INK_SOFT} !important;
  padding: 7px 14px;
}}
:host .bk-btn:hover {{ background: {SURFACE_INSET} !important; color: {INK} !important; border-color: {BORDER_STRONG} !important; }}
"""
)

BUTTON_CLOSE = """
:host .bk-btn {
  background: transparent !important;
  border: 1px solid #E2E8F0 !important;
  color: #9AA7B4 !important;
  border-radius: 8px;
  padding: 7px 11px;
  transition: all .15s ease;
}
:host .bk-btn:hover { background: #FEF2F2 !important; color: #D92D20 !important; border-color: #FCA5A5 !important; }
"""

# The card host itself IS the `.card` element, so its box (border/radius/shadow) is styled
# via :host. Inside the shadow root the header is a <button class="card-header"> followed by
# the body widgets as flat siblings (no body wrapper), so `:host > div` pads exactly the body
# content (the header button is excluded) to give the fields breathing room from the edge.

# Big sectioning cards (Session & Subject, Channels, Advanced YAML, action footer).
SECTION_CARD = f"""
:host {{
  border: 1px solid {BORDER} !important;
  border-radius: 14px !important;
  background: {SURFACE} !important;
  box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 6px 16px rgba(16,24,40,.05) !important;
  overflow: hidden;
}}
:host .card-header {{
  background: {SURFACE} !important;
  border-bottom: 1px solid #EEF2F6 !important;
  padding: 15px 26px !important;
}}
:host .card-title, :host .card-title * {{
  color: {INK} !important;
  font-weight: {SEMIBOLD} !important;
  font-size: {HEADING_SIZE} !important;
}}
:host .card-button, :host .card-header svg {{ color: #7A8896 !important; fill: #7A8896 !important; }}
:host > div {{ padding-left: 26px; padding-right: 26px; }}
:host > div:first-of-type {{ padding-top: 16px; }}
:host > div:last-child {{ padding-bottom: 22px; }}
"""

# Lighter tinted cards for the items inside a section: each device category (collapsible) and
# each fiber-photometry channel (header-less). They read as distinct items on the white section.
CATEGORY_CARD = f"""
:host {{
  border: 1px solid #E6EBF1 !important;
  border-radius: 10px !important;
  background: {SURFACE_INSET} !important;
  box-shadow: none !important;
  overflow: hidden;
}}
:host .card-header {{
  background: {SURFACE_INSET} !important;
  border-bottom: 1px solid transparent !important;
  padding: 11px 18px !important;
}}
:host(:hover) .card-header {{ background: #EEF3F8 !important; }}
:host .card-title, :host .card-title * {{
  color: {INK} !important;
  font-weight: {SEMIBOLD} !important;
  font-size: {BODY_SIZE} !important;
}}
:host .card-button, :host .card-header svg {{ color: #8A97A4 !important; fill: #8A97A4 !important; }}
:host > div {{ padding-left: 18px; padding-right: 18px; }}
:host > div:first-of-type {{ padding-top: 14px; }}
:host > div:last-child {{ padding-bottom: 16px; }}
"""

# Each device instance inside a category: a white hairline item.
INSTANCE_CARD = f"""
:host {{
  border: 1px solid {BORDER} !important;
  border-radius: 9px !important;
  background: {SURFACE} !important;
  box-shadow: 0 1px 1px rgba(16,24,40,.03) !important;
  overflow: hidden;
}}
:host .card-header {{
  background: {SURFACE} !important;
  border-bottom: 1px solid #F1F4F8 !important;
  padding: 10px 18px !important;
}}
:host .card-title, :host .card-title * {{
  color: {INK} !important;
  font-weight: {SEMIBOLD} !important;
  font-size: {BODY_SIZE} !important;
}}
:host .card-button, :host .card-header svg {{ color: #9AA7B4 !important; fill: #9AA7B4 !important; }}
:host > div {{ padding-left: 18px; padding-right: 18px; }}
:host > div:first-of-type {{ padding-top: 14px; }}
:host > div:last-child {{ padding-bottom: 16px; }}
"""

# Document-level chrome: page background, font smoothing, template header polish.
DOCUMENT_CSS = f"""
body {{ background: {PAGE} !important; }}
.bk-root, body {{ -webkit-font-smoothing: antialiased; }}
#header, .pn-bar, .mdc-top-app-bar {{ box-shadow: 0 1px 0 rgba(16,24,40,.06); }}
::-webkit-scrollbar {{ width: 11px; height: 11px; }}
::-webkit-scrollbar-thumb {{ background: #CBD5E1; border-radius: 6px; border: 3px solid {PAGE}; }}
::-webkit-scrollbar-thumb:hover {{ background: #AEBAC8; }}
"""

# Radio group (age / date-of-birth selector): tighten and tint.
RADIO_STYLESHEET = f"""
:host label {{ color: {INK}; font-weight: {REGULAR}; font-size: {BODY_SIZE}; }}
:host input[type="radio"] {{ accent-color: {ACCENT}; }}
"""

# Alert panes (success / danger): flatter, rounded, tinted.
ALERT_STYLESHEET = f"""
:host .alert {{ border-radius: 10px; border: 1px solid {BORDER}; font-size: {BODY_SIZE}; }}
:host .alert-success {{ background: #ECFDF3; border-color: #ABEFC6; color: #1E7B43; }}
:host .alert-danger {{ background: #FEF3F2; border-color: #FECDCA; color: #B42318; }}
"""

# ----------------------------------------------------------------------------------------------------------------------
# Label humanizing
# ----------------------------------------------------------------------------------------------------------------------
_ACRONYMS = {
    "nm": "nm",
    "na": "NA",
    "ap": "AP",
    "ml": "ML",
    "dv": "DV",
    "um": "µm",
    "mm": "mm",
    "nl": "nL",
    "id": "ID",
    "w": "W",
    "rrid": "RRID",
    "doi": "DOI",
}

# Keys that end in "_in_<unit>" carry that unit, which the label shows in parentheses.
_UNITS = {"nm", "um", "mm", "nl", "w"}


def humanize(name: str) -> str:
    """Turn a snake_case metadata key into a Title Case field label, with a trailing unit in parentheses."""
    words = name.split("_")
    unit = ""
    if len(words) > 2 and words[-2] == "in" and words[-1] in _UNITS:
        unit = f" ({_ACRONYMS[words[-1]]})"
        words = words[:-2]
    return title_case(" ".join(_ACRONYMS.get(word, word) for word in words)) + unit


# ----------------------------------------------------------------------------------------------------------------------
# Inline-styled HTML chrome (Markdown panes; inline styles bypass the shadow boundary)
# ----------------------------------------------------------------------------------------------------------------------
def _markdown(html: str, **styles: str) -> pn.pane.Markdown:
    return pn.pane.Markdown(html, sizing_mode="stretch_width", styles=styles, margin=(0, 0))


def page_header(title: str, subtitle: str) -> pn.pane.Markdown:
    """A page-top title block: name + one-line subtitle over a thin accent rule."""
    html = (
        f"<div style='padding:4px 0 2px 0;'>"
        f"<div style='font-size:{HEADING_SIZE};font-weight:{SEMIBOLD};color:{INK};'>{title}</div>"
        f"<div style='font-size:{BODY_SIZE};color:{INK};margin-top:3px;'>{subtitle}</div>"
        f"<div style='height:3px;width:46px;background:{ACCENT};border-radius:3px;margin-top:12px;'></div>"
        f"</div>"
    )
    return _markdown(html)


def section_label(text: str, note: str = "") -> pn.pane.Markdown:
    """A section label with an optional trailing note, over a hairline rule."""
    note_html = (
        f"<span style='font-weight:{REGULAR};color:{INK_SOFT};font-size:{SMALL_SIZE};margin-left:10px;'>{note}</span>"
        if note
        else ""
    )
    html = (
        f"<div style='display:flex;align-items:baseline;border-bottom:1px solid {BORDER};padding-bottom:7px;margin:6px 0 2px 0;'>"
        f"<span style='font-size:{BODY_SIZE};font-weight:{SEMIBOLD};color:{INK};'>{text}</span>"
        f"{note_html}</div>"
    )
    return _markdown(html)


def subgroup_label(text: str) -> pn.pane.Markdown:
    """A quieter label that groups several device categories (e.g. Hardware / Biology)."""
    html = (
        f"<div style='font-size:{BODY_SIZE};font-weight:{SEMIBOLD};color:{INK};margin:14px 2px 2px 2px;'>{text}</div>"
    )
    return _markdown(html)


def help_note(text: str) -> pn.pane.Markdown:
    """Muted helper text under a heading."""
    return _markdown(f"<div style='font-size:{SMALL_SIZE};color:{INK_SOFT};line-height:1.45;'>{text}</div>")


def intro_note(html: str) -> pn.pane.Markdown:
    """A short page-top instruction paragraph (slightly more present than help_note)."""
    return _markdown(
        f"<div style='font-size:{BODY_SIZE};color:{INK};line-height:1.45;max-width:780px;margin:2px 0 4px 0;'>{html}</div>"
    )


def channel_role_chip(role: str, store_name: str) -> pn.pane.Markdown:
    """A header for one channel instance: role pill + the source store name (recording site is the group)."""
    role_color = ACCENT if role == "signal" else "#7A8896"
    role_bg = ACCENT_SOFT if role == "signal" else "#EEF2F6"
    html = (
        f"<div style='display:flex;align-items:center;gap:10px;'>"
        f"<span style='font-size:{SMALL_SIZE};font-weight:{SEMIBOLD};color:{role_color};background:{role_bg};"
        f"padding:2px 11px;border-radius:20px;text-transform:capitalize;'>{role}</span>"
        f"<span style='font-size:{SMALL_SIZE};color:{INK_SOFT};font-family:{MONOSPACE_FAMILY};'>{store_name}</span>"
        f"</div>"
    )
    return _markdown(html)
