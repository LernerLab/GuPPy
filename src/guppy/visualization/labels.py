"""How plot axes and titles name time and the traces GuPPy writes, in Title Case."""

# The time axis every trace and PSTH is plotted against. Plots that share it link their x-axes.
TIME_LABEL = "Time (s)"

# The file prefix of each trace GuPPy writes for a recording site, and the name a plot gives it.
_TRACE_NAMES = {
    "cntrl_sig_fit_": "Control Fit",
    "control_": "Control",
    "signal_": "Signal",
    "z_score_": "Z-Score",
    "dff_": "ΔF/F",
}


def trace_title(trace: str) -> str:
    """Name a recording site's trace by what it is, followed by the site.

    Parameters
    ----------
    trace : str
        The trace's file name without extension, such as ``"z_score_dms"`` or ``"control_dms"``.

    Returns
    -------
    str
        The trace's name and its recording site, such as ``"Z-Score (dms)"``.
    """
    prefix = next(prefix for prefix in _TRACE_NAMES if trace.startswith(prefix))
    return f"{_TRACE_NAMES[prefix]} ({trace[len(prefix):]})"
