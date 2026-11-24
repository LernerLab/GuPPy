"""Tile toggles: a choice between a few options, each drawn as a tile with an icon, a name, and a description."""

from dataclasses import dataclass

from .header import LOGO_BLUE, LOGO_VIOLET
from .typography import INK, SECONDARY_INK, SMALL_SIZE

TILE_BORDER = "#DEE2E6"

# Tabler icon outlines, drawn as masks so each takes the color the stylesheet gives it.
FOLDER_ICON = "M5 4h4l3 3h7a2 2 0 0 1 2 2v8a2 2 0 0 1 -2 2h-14a2 2 0 0 1 -2 -2v-11a2 2 0 0 1 2 -2"
CLOUD_ICON = (
    "M6.657 18c-2.572 0 -4.657 -2.007 -4.657 -4.483c0 -2.475 2.085 -4.482 4.657 -4.482c.393 -1.762 1.794 -3.2 "
    "3.675 -3.773c1.88 -.572 3.956 -.193 5.444 1c1.488 1.19 2.162 3.007 1.77 4.769h.99c1.913 0 3.464 1.56 3.464 "
    "3.486c0 1.927 -1.551 3.487 -3.465 3.487h-11.878"
)
PEN_ICON = "M4 20h4l10.5 -10.5a2.828 2.828 0 1 0 -4 -4l-10.5 10.5v4M13.5 6.5l4 4"
HAND_ICON = (
    "M8 13v-7.5a1.5 1.5 0 0 1 3 0v6.5M11 5.5v-2a1.5 1.5 0 1 1 3 0v8.5M14 5.5a1.5 1.5 0 0 1 3 0v6.5"
    "M17 7.5a1.5 1.5 0 0 1 3 0v8.5a6 6 0 0 1 -6 6h-2h.208a6 6 0 0 1 -5.012 -2.7l-.196 -.3c-.312 -.479 "
    "-1.407 -2.388 -3.286 -5.728a1.5 1.5 0 0 1 .536 -2.022a1.867 1.867 0 0 1 2.28 .28l1.47 1.47"
)


@dataclass(frozen=True)
class Tile:
    """How one option of a tile toggle is drawn.

    Parameters
    ----------
    icon : str
        SVG path data for the icon, on a 24-unit grid.
    accent : str
        The color the tile takes while selected.
    description : str or None
        One line under the option's name, or ``None`` for a compact toggle, which shows none.
    """

    icon: str
    accent: str
    description: str | None = None


LOCAL_FOLDER_TILE = Tile(icon=FOLDER_ICON, accent=LOGO_BLUE, description="Session folders on this computer")
DANDI_ARCHIVE_TILE = Tile(icon=CLOUD_ICON, accent=LOGO_VIOLET, description="Stream NWB sessions from dandiarchive.org")


def _icon_mask(*, path: str) -> str:
    svg = (
        "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' "
        f"stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='{path}'/%3E%3C/svg%3E"
    )
    return f'url("data:image/svg+xml;utf8,{svg}")'


def tile_toggle_stylesheet(*, tiles: list[Tile], compact: bool = False) -> str:
    """Return the stylesheet that draws a RadioButtonGroup's options as equal-width tiles.

    Parameters
    ----------
    tiles : list of Tile
        One per option, in the group's option order.
    compact : bool, default False
        Draw one-line tiles, icon and name only, at the height of a dropdown so the toggle
        sits in a toolbar row beside them.

    Returns
    -------
    str
        CSS for the group's ``stylesheets``.
    """
    per_tile = "".join(
        f"""
:host .bk-btn-group > button:nth-child({position}) {{
  --icon: {_icon_mask(path=tile.icon)};
  --accent: {tile.accent};
  --description: "{tile.description or ''}";
}}"""
        for position, tile in enumerate(tiles, start=1)
    )
    return (_COMPACT_TILES_CSS if compact else _TILES_CSS) + per_tile


_SHARED_TILE_CSS = f"""
:host .bk-btn-group > button {{
  flex: 1 1 0 !important;
  background: #FFFFFF !important;
  color: {INK} !important;
  font-weight: 600;
  border: 1px solid {TILE_BORDER} !important;
  box-shadow: none !important;
}}
:host .bk-btn-group > button::before {{
  content: "";
  background-color: {SECONDARY_INK};
  -webkit-mask: var(--icon) no-repeat center / contain;
  mask: var(--icon) no-repeat center / contain;
}}
:host .bk-btn-group > button:hover {{
  border-color: var(--accent) !important;
}}
:host .bk-btn-group > button.bk-active {{
  background: color-mix(in srgb, var(--accent) 7%, white) !important;
}}
:host .bk-btn-group > button.bk-active::before {{
  background-color: var(--accent);
}}
"""

# Matches a dropdown's 35px height and 6px corners. The selected tile's heavier edge is an inset
# shadow so selecting it does not shift the label.
_COMPACT_TILES_CSS = (
    _SHARED_TILE_CSS
    + """
:host .bk-btn-group {
  gap: 8px;
}
:host .bk-btn-group > button {
  display: flex !important;
  align-items: center;
  justify-content: center;
  gap: 8px;
  height: 35px !important;
  padding: 0 12px !important;
  border-radius: 6px !important;
}
:host .bk-btn-group > button::before {
  width: 16px;
  height: 16px;
}
:host .bk-btn-group > button.bk-active {
  border-color: var(--accent) !important;
  box-shadow: inset 0 0 0 1px var(--accent) !important;
}
"""
)

_TILES_CSS = (
    _SHARED_TILE_CSS
    + f"""
:host .bk-btn-group {{
  gap: 16px;
}}
:host .bk-btn-group > button {{
  display: grid !important;
  grid-template-columns: 28px 1fr;
  column-gap: 12px;
  align-items: center;
  text-align: left;
  height: auto !important;
  padding: 14px 18px !important;
  border-radius: 10px !important;
}}
:host .bk-btn-group > button::before {{
  grid-row: 1 / span 2;
  width: 28px;
  height: 28px;
}}
:host .bk-btn-group > button::after {{
  content: var(--description);
  grid-column: 2;
  font-size: {SMALL_SIZE};
  font-weight: 400;
  color: {SECONDARY_INK};
}}
:host .bk-btn-group > button.bk-active {{
  border: 2px solid var(--accent) !important;
  padding: 13px 17px !important;
}}
"""
)
