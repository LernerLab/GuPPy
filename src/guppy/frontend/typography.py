"""GuPPy's type system: one text family, one monospace for literals, four sizes and two weights.

Panel renders every component in its own shadow root, which page-level CSS cannot reach, so the
rules ride on a design whose stylesheet Panel adds to every component.
"""

from typing import ClassVar

from panel.theme.base import Inherit
from panel.theme.bootstrap import Bootstrap
from panel.viewable import Viewable

TEXT_FAMILY = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
MONOSPACE_FAMILY = 'ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace'

TITLE_SIZE = "20px"
HEADING_SIZE = "16px"
BODY_SIZE = "14px"
SMALL_SIZE = "12px"

REGULAR = 400
SEMIBOLD = 600

INK = "#212529"
SECONDARY_INK = "#6C757D"

# The class a component opts into to be set as help or status text.
SMALL_CLASS = "small"

COMPONENT_CSS = f"""
:host {{
  --bokeh-base-font: {TEXT_FAMILY};
  --bokeh-mono-font: {MONOSPACE_FAMILY};
  --bokeh-font-size: {BODY_SIZE};
  --bs-font-sans-serif: {TEXT_FAMILY};
  --bs-font-monospace: {MONOSPACE_FAMILY};
  --bs-body-font-size: {BODY_SIZE};
  --bs-btn-font-size: {BODY_SIZE};
  font-family: {TEXT_FAMILY};
  font-size: {BODY_SIZE};
  line-height: 1.45;
  color: {INK};
}}
input, select, textarea, option, button, .bk-input, .bk-btn, .bk-tab, label, .tabulator {{
  font-family: {TEXT_FAMILY};
  font-size: {BODY_SIZE};
}}
label {{
  font-weight: {REGULAR};
}}
input, .tabulator-cell {{
  font-variant-numeric: tabular-nums;
}}
.tabulator .tabulator-header .tabulator-col .tabulator-col-title {{
  font-size: {BODY_SIZE};
  font-weight: {SEMIBOLD};
}}
.tabulator .tabulator-tableholder .tabulator-table .tabulator-row .tabulator-cell {{
  font-size: {BODY_SIZE};
}}
b, strong, th {{
  font-weight: {SEMIBOLD};
}}
h1, h2 {{
  font-size: {TITLE_SIZE};
  font-weight: {SEMIBOLD};
  line-height: 1.25;
}}
h3 {{
  font-size: {BODY_SIZE};
  font-weight: {SEMIBOLD};
  line-height: 1.25;
}}
h4, h5, h6 {{
  font-size: {BODY_SIZE};
  font-weight: {SEMIBOLD};
}}
p {{
  font-size: {BODY_SIZE};
}}
/* Monospace looks larger than the text face at the same size, so it is set a little smaller. */
code, pre, kbd {{
  font-family: {MONOSPACE_FAMILY};
  font-size: 0.93em;
}}
.card-header, :host(.card-title) h3 {{
  font-size: {HEADING_SIZE};
  font-weight: {SEMIBOLD};
}}
:host(.{SMALL_CLASS}), :host(.{SMALL_CLASS}) p {{
  font-size: {SMALL_SIZE};
  color: {SECONDARY_INK};
}}
"""


class GuppyTypeDesign(Bootstrap):
    """Bootstrap with GuPPy's type system applied inside every component."""

    modifiers: ClassVar[dict] = {Viewable: {"stylesheets": [Inherit, COMPONENT_CSS]}}
