"""Display-time downsampling for the long time-series traces the pipeline plots.

A full-length recording holds millions of samples per trace, which Bokeh would ship to
the browser in full and render point by point. Wrapping a curve in
:func:`downsample_for_display` keeps the whole array server-side and sends only the
samples the plot is wide enough to show, re-derived from the full array whenever the
user zooms.
"""

import holoviews as hv
from holoviews.operation.downsample import downsample1d

# Pixel width shared by every trace plot, and therefore the number of samples worth
# sending: the downsampler resolves one bucket per pixel column.
PLOT_WIDTH = 750


def downsample_for_display(curve: hv.Curve) -> hv.DynamicMap:
    """Wrap a curve so only the currently visible range is sent to the browser.

    The returned object composes like the curve it wraps — it can be overlaid, laid out,
    and given options as usual — but resolves to a downsampled curve at render time and
    re-resolves against the full data on every zoom or pan.

    The ``m4`` algorithm keeps the minimum, maximum, first and last sample of each pixel
    column, which is what makes the drawn line identical to plotting every sample. A
    one-sample artifact spike therefore stays visible at any zoom level.

    Parameters
    ----------
    curve : hv.Curve
        Curve over the full-resolution trace.

    Returns
    -------
    hv.DynamicMap
        Range-linked view of ``curve``.
    """
    return downsample1d(curve, algorithm="m4", width=PLOT_WIDTH, dynamic=True)
