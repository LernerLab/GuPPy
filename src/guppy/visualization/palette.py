"""GuPPy's plot colors, named by the role each one plays in a plot, and its plot text.

The signal and everything derived from it is green, the color most photometry indicators fluoresce; the
isosbestic control is blue. Covariates are violet, and red marks what should stand out from the data, such
as significant windows and transient peaks.

Plot text is set in Helvetica or Arial, which every machine has, so a saved SVG keeps its type wherever it is
opened. Its sizes are chosen for a saved figure, 1000 pixels wide by default, placed at the full width of a
journal page: 16-pixel titles and axis labels land near 8 points there, and 14-pixel ticks and legends near 7.
"""

import holoviews as hv
from bokeh.themes import Theme

BLUE = "#1F6FD1"
VIOLET = "#7B4FD0"
GREEN = "#3A9E5C"
RED = "#D62728"
AMBER = "#D08C15"
CYAN = "#1598A0"
GRAY = "#6B7785"
REFERENCE_LINE = "#8A96A3"

SIGNAL = GREEN
CONTROL = BLUE
COVARIATE = VIOLET
HIGHLIGHT = RED

# Tonic epoch windows, in the order the epochs are listed, sampled along the GuPPy logo's gradient
# from violet through blue and teal to the yellow-green at the tip of its tail.
EPOCH_COLORS = ["#6E56CF", "#1F7FC4", "#14A39A", "#9BC53D"]

# Categorical plot colors, in the order a plot with several series takes them.
PLOT_COLORS = [GREEN, BLUE, VIOLET, AMBER, CYAN, RED, GRAY]

# Registered under its own name so a plot can offer the palette by name, alongside Bokeh's palettes.
hv.Cycle.default_cycles["GuPPy"] = PLOT_COLORS

PLOT_FONT = "Helvetica, Arial, sans-serif"
PLOT_LABEL_SIZE = "16px"
PLOT_TICK_SIZE = "14px"

PLOT_THEME = Theme(
    json={
        "attrs": {
            "Title": {"text_font": PLOT_FONT, "text_font_size": PLOT_LABEL_SIZE, "text_font_style": "normal"},
            "Axis": {
                "axis_label_text_font": PLOT_FONT,
                "axis_label_text_font_size": PLOT_LABEL_SIZE,
                "axis_label_text_font_style": "normal",
                "major_label_text_font": PLOT_FONT,
                "major_label_text_font_size": PLOT_TICK_SIZE,
            },
            "Legend": {
                "label_text_font": PLOT_FONT,
                "label_text_font_size": PLOT_TICK_SIZE,
                "title_text_font": PLOT_FONT,
                "title_text_font_size": PLOT_TICK_SIZE,
            },
            "ColorBar": {
                "title_text_font": PLOT_FONT,
                "title_text_font_size": PLOT_LABEL_SIZE,
                "title_text_font_style": "normal",
                "major_label_text_font": PLOT_FONT,
                "major_label_text_font_size": PLOT_TICK_SIZE,
            },
        }
    }
)


def apply_plot_defaults() -> None:
    """Make the palette and plot text the default for every HoloViews plot that does not choose its own.

    A plot of one series takes the first palette color, the signal's green; an overlay of several steps through
    the palette in order.
    """
    palette_cycle = hv.Cycle(PLOT_COLORS)
    hv.opts.defaults(
        hv.opts.Curve(color=palette_cycle),
        hv.opts.Scatter(color=palette_cycle),
        hv.opts.Bars(color=palette_cycle),
        hv.opts.Spread(color=palette_cycle),
        hv.opts.Area(color=palette_cycle),
        hv.opts.Histogram(color=palette_cycle),
        hv.opts.HLine(color=REFERENCE_LINE),
        hv.opts.VLine(color=REFERENCE_LINE),
    )
    hv.renderer("bokeh").theme = PLOT_THEME
