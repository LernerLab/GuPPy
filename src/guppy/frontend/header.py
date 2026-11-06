"""The GuPPy header: a white bar with the GuPPy mark, the wordmark in the logo's gradient, and a rainbow rule."""

from pathlib import Path

import panel as pn

ASSETS_DIRECTORY = Path(__file__).parent.parent / "assets"

# The GuPPy logo's gradient, violet through blue and teal to green.
LOGO_GRADIENT = "linear-gradient(90deg, #6E56CF, #1F7FC4, #14A39A, #5DB45C)"

HEADER_BACKGROUND = "#FFFFFF"
HEADER_TEXT_COLOR = "#212529"
_SUBTITLE_COLOR = "#495057"
_TOGGLE_ICON = (
    "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 30 30'%3E"
    "%3Cpath stroke='%23495057' stroke-width='2' stroke-linecap='round' d='M4 7h22M4 15h22M4 23h22'/%3E%3C/svg%3E\")"
)

# The template draws its title as plain text; it is hidden so the wordmark can stand in its place,
# while the title still names the browser tab.
HEADER_CSS = f"""
#header {{
  box-shadow: none !important;
  border-bottom: 3px solid transparent;
  border-image: {LOGO_GRADIENT} 1;
}}
#header .app-header .title {{
  display: none;
}}
#header .app-logo {{
  height: 34px;
}}
#header.navbar {{
  --bs-navbar-toggler-icon-bg: {_TOGGLE_ICON};
}}
"""


def wordmark(*, page_name: str) -> pn.pane.HTML:
    """Return the header's wordmark: "GuPPy" in the logo's gradient, followed by the page's name.

    Parameters
    ----------
    page_name : str
        Shown beside the wordmark.

    Returns
    -------
    panel.pane.HTML
        The wordmark, for the template's header.
    """
    return pn.pane.HTML(
        f'<span style="font-weight: 800; font-size: 24px; background: {LOGO_GRADIENT}; '
        f'-webkit-background-clip: text; background-clip: text; color: transparent;">GuPPy</span>'
        f'<span style="font-size: 16px; color: {_SUBTITLE_COLOR}; margin-left: 12px;">{page_name}</span>',
        margin=(0, 4),
        align="center",
    )


def guppy_header_template(*, page_name: str, title: str) -> pn.template.BootstrapTemplate:
    """Build a page template with the GuPPy header: white, with the mark, the wordmark, and a rainbow rule.

    Parameters
    ----------
    page_name : str
        Shown in the header beside the wordmark.
    title : str
        The browser tab's title.

    Returns
    -------
    panel.template.BootstrapTemplate
        The template, with the wordmark already in its header.
    """
    template = pn.template.BootstrapTemplate(
        title=title,
        logo=str(ASSETS_DIRECTORY / "GuppyMark.png"),
        favicon=str(ASSETS_DIRECTORY / "favicon.png"),
        header_background=HEADER_BACKGROUND,
        header_color=HEADER_TEXT_COLOR,
    )
    template.config.raw_css = [*template.config.raw_css, HEADER_CSS]
    template.header.append(wordmark(page_name=page_name))
    return template
