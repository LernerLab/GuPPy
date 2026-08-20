"""Display-time rendering for the long time-series traces the pipeline plots.

A full-length recording holds millions of samples per trace, which Bokeh would ship to the
browser in full and render point by point. :func:`shade_trace` keeps the whole array
server-side and sends only an image of what the plot is wide enough to show, re-aggregated
from the full array whenever the user zooms.

It renders the density of samples per pixel rather than a reduced line, because density is
what separates noise from signal here: with millions of samples in a few hundred pixel
columns, the minimum and maximum of a column span the whole noise band, so a line through
them gives the band and a brief transient identical weight. Counting samples per pixel
instead leaves the noise band a solid mass and a transient a sparse, sharp excursion above
it — the distinction that makes an artifact visible.
"""

import holoviews as hv
from holoviews.operation.datashader import datashade

# Hues for a shaded trace. datashade ramps opacity from the background up to the color with
# the number of samples in the pixel, so one color carries the whole density scale.
TRACE_COLOR = "#1f77b4"
FIT_COLOR = "#ff7f0e"


def shade_trace(curve: hv.Curve, color: str = TRACE_COLOR) -> hv.DynamicMap:
    """Render a trace by per-pixel sample density.

    The returned object composes like the curve it wraps and re-aggregates against the full
    data on every zoom or pan, so no sample is ever dropped — a one-sample artifact spike
    remains visible at any zoom level.

    Parameters
    ----------
    curve : hv.Curve
        Curve over the full-resolution trace.
    color : str, optional
        Hue to shade the trace in, for telling overlaid traces apart.

    Returns
    -------
    hv.DynamicMap
        Range-linked density image of ``curve``.
    """
    return datashade(curve, cmap=[color], line_width=1)
