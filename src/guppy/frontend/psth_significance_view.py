"""Step-5 dashboard tab for PSTH significance comparisons.

Reads the ``significance_<comparison>.h5`` tables written into
``psth_significance_output/`` and plots the selected comparison. Output directories
analysed without the significance parameter hold no such tables, so the tab renders a
short note instead and can be added to the dashboard unconditionally.

Serves session run folders and group folders alike: they hold the same files, and only
the caption differs, since a group's comparison resamples session averages rather than
trials.
"""

import glob
import logging
import os

import holoviews as hv
import panel as pn

from ..analysis.io_utils import PSTH_SIGNIFICANCE_DIRNAME, PSTH_SIGNIFICANCE_PREFIX
from ..analysis.standard_io import read_psth_significance_from_hdf5
from ..visualization.psth_significance import build_significance_panel

logger = logging.getLogger(__name__)


def significance_comparisons(filepath: str) -> list[str]:
    """Return the comparison names ``filepath`` holds significance results for, sorted.

    Parameters
    ----------
    filepath : str
        Output directory: a session run folder or a group folder.

    Returns
    -------
    list of str
        Comparison names, empty when the directory has no significance results.
    """
    pattern = os.path.join(filepath, PSTH_SIGNIFICANCE_DIRNAME, PSTH_SIGNIFICANCE_PREFIX + "*.h5")
    return sorted(os.path.basename(path)[len(PSTH_SIGNIFICANCE_PREFIX) : -len(".h5")] for path in glob.glob(pattern))


def describe_comparison(*, name: str, n: int, n_b: int | None) -> str:
    """Build the caption naming what a comparison tested and over how many samples.

    Parameters
    ----------
    name : str
        Comparison name, as it appears in the filename.
    n : int
        Number of resampled columns on the first side.
    n_b : int or None
        Number of resampled columns on the second side, or None for a test against zero.

    Returns
    -------
    str
        Markdown caption.
    """
    if n_b is None:
        return f"**{name}** tested against zero, resampling {n} trials or session averages."
    return (
        f"**{name}**, resampling {n} and {n_b} trials or session averages. "
        f"Positive values mean the first event is larger."
    )


class PsthSignificanceView:
    """A comparison selector over the significance results of one output directory.

    Parameters
    ----------
    filepath : str
        Output directory holding a ``psth_significance_output/`` subdirectory.
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.results_path = os.path.join(filepath, PSTH_SIGNIFICANCE_DIRNAME)
        self.comparisons = significance_comparisons(filepath)

        self.comparison_select = pn.widgets.Select(
            name="Comparison", options=self.comparisons, value=self.comparisons[0], width=520
        )
        self.caption = pn.pane.Markdown(self._make_caption(), width=520)
        self.plot_pane = pn.pane.HoloViews(self._make_plot(), sizing_mode="stretch_width")

        self.comparison_select.param.watch(self._on_comparison_change, "value")

    def _read(self) -> object:
        return read_psth_significance_from_hdf5(filepath=self.results_path, name=self.comparison_select.value)

    def _make_caption(self) -> str:
        significance = self._read()
        n_b = int(significance["n_b"].iloc[0]) if "n_b" in significance.columns else None
        return describe_comparison(name=self.comparison_select.value, n=int(significance["n"].iloc[0]), n_b=n_b)

    def _make_plot(self) -> hv.Overlay:
        significance = self._read()
        is_difference = "n_b" in significance.columns
        value_label = "difference (z-score or ΔF/F)" if is_difference else "z-score or ΔF/F"

        return build_significance_panel(
            timestamps=significance["timestamps"].to_numpy(),
            estimate=significance["estimate"].to_numpy(),
            ci_lower=significance["ci_lower"].to_numpy(),
            ci_upper=significance["ci_upper"].to_numpy(),
            significant=significance["significant"].to_numpy(),
            value_label=value_label,
            title=self.comparison_select.value,
        )

    def _on_comparison_change(self, event: object) -> None:
        self.caption.object = self._make_caption()
        self.plot_pane.object = self._make_plot()

    @property
    def widget(self) -> pn.Column:
        """The composed selector, caption and plot."""
        return pn.Column(self.comparison_select, self.caption, self.plot_pane)


def build_psth_significance_view(filepath: str) -> pn.Column:
    """Build the Significance tab for one output directory.

    Parameters
    ----------
    filepath : str
        Output directory: a session run folder or a group folder.

    Returns
    -------
    pn.Column
        The selector and plot, or a short note when the directory was analysed without
        significance testing.
    """
    if not significance_comparisons(filepath):
        return pn.Column(pn.pane.Markdown("_No PSTH significance results in this session._"))

    return PsthSignificanceView(filepath).widget
