import pytest

from guppy.visualization.labels import TIME_LABEL, trace_title


class TestTraceTitle:
    @pytest.mark.parametrize(
        ("trace", "title"),
        [
            ("control_dms", "Control (dms)"),
            ("signal_dms", "Signal (dms)"),
            ("cntrl_sig_fit_dms", "Control Fit (dms)"),
            ("z_score_DLS", "Z-Score (DLS)"),
            ("dff_region", "ΔF/F (region)"),
        ],
    )
    def test_names_the_trace_and_its_site(self, trace, title):
        assert trace_title(trace) == title

    def test_a_site_with_underscores_keeps_them(self):
        assert trace_title("z_score_left_dms") == "Z-Score (left_dms)"


def test_time_label_is_title_case():
    assert TIME_LABEL == "Time (s)"
