import numpy as np
import pandas as pd
import pytest

from guppy.analysis.group_tables import stack_member_tables, summarize_member_tables


@pytest.fixture
def member_tables():
    """Two members' tonic means over the same two epochs, listed in different orders."""
    first = pd.DataFrame(
        {"mean_zscore": [1.0, 3.0], "mean_dff": [0.5, np.nan]},
        index=pd.Index(["baseline", "drug"], name="epoch"),
    )
    second = pd.DataFrame(
        {"mean_zscore": [7.0, 2.0], "mean_dff": [0.5, 1.5]},
        index=pd.Index(["drug", "baseline"], name="epoch"),
    )
    return {"subject1/session1/output_1": first, "subject2/session1/output_1": second}


class TestStackMemberTables:
    def test_prepends_member_level(self, member_tables):
        stacked = stack_member_tables(member_tables=member_tables)

        assert list(stacked.index.names) == ["member", "epoch"]
        assert list(stacked.index) == [
            ("subject1/session1/output_1", "baseline"),
            ("subject1/session1/output_1", "drug"),
            ("subject2/session1/output_1", "drug"),
            ("subject2/session1/output_1", "baseline"),
        ]
        np.testing.assert_array_equal(stacked["mean_zscore"].to_numpy(), [1.0, 3.0, 7.0, 2.0])


class TestSummarizeMemberTables:
    def test_mean_sem_and_count(self, member_tables):
        summary = summarize_member_tables(member_tables=member_tables, value_columns=["mean_zscore", "mean_dff"])

        assert list(summary.index) == ["baseline", "drug"]
        # baseline: (1, 2) -> mean 1.5, sample std 0.7071, sem 0.5; drug: (3, 7) -> mean 5, sem 2
        np.testing.assert_allclose(summary["mean_zscore_mean"].to_numpy(), [1.5, 5.0])
        np.testing.assert_allclose(summary["mean_zscore_sem"].to_numpy(), [0.5, 2.0])
        np.testing.assert_array_equal(summary["mean_zscore_n"].to_numpy(), [2, 2])

    def test_nan_member_is_left_out(self, member_tables):
        summary = summarize_member_tables(member_tables=member_tables, value_columns=["mean_dff"])

        # baseline: (0.5, 1.5) -> mean 1.0, sem 0.5; drug: only 0.5 survives, so its sem is NaN
        np.testing.assert_allclose(summary["mean_dff_mean"].to_numpy(), [1.0, 0.5])
        np.testing.assert_allclose(summary["mean_dff_sem"].to_numpy(), [0.5, np.nan])
        np.testing.assert_array_equal(summary["mean_dff_n"].to_numpy(), [2, 1])

    def test_multi_level_index(self):
        index = pd.MultiIndex.from_tuples(
            [("mean_zscore", "akinesia"), ("mean_dff", "akinesia")], names=["metric", "covariate"]
        )
        member_tables = {
            "run_a": pd.DataFrame({"pearson_r": [0.2, 0.4]}, index=index),
            "run_b": pd.DataFrame({"pearson_r": [0.6, 0.8]}, index=index),
        }

        summary = summarize_member_tables(member_tables=member_tables, value_columns=["pearson_r"])

        assert list(summary.index) == [("mean_zscore", "akinesia"), ("mean_dff", "akinesia")]
        np.testing.assert_allclose(summary["pearson_r_mean"].to_numpy(), [0.4, 0.6])
        np.testing.assert_allclose(summary["pearson_r_sem"].to_numpy(), [0.2, 0.2])
