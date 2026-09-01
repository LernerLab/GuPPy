import os

import numpy as np
import pandas as pd
import pytest

from guppy.orchestration.psth_significance import (
    comparison_name,
    compute_comparison,
    plan_comparisons,
    psth_metrics,
    read_psth_samples,
)


def write_stores_list(run_folder, store_ids, store_labels):
    np.savetxt(
        os.path.join(run_folder, "storesList.csv"),
        np.asarray([store_ids, store_labels]),
        delimiter=",",
        fmt="%s",
    )


def write_psth(run_folder, event, recording_site, basename, *, num_trials=5, num_timepoints=6, bins=False):
    """Write a PSTH file with the layout create_Df_for_psth produces."""
    columns, data = [], []
    for trial in range(num_trials):
        columns.append(str(100.0 + trial))
        data.append(np.full(num_timepoints, float(trial)))
    if bins:
        columns += ["bin_(0-2)", "bin_err_(0-2)"]
        data += [np.zeros(num_timepoints), np.zeros(num_timepoints)]
    columns += ["timestamps", "mean", "err"]
    data += [np.linspace(-1, 1, num_timepoints), np.zeros(num_timepoints), np.zeros(num_timepoints)]

    frame = pd.DataFrame(np.asarray(data).T, columns=columns, dtype="float32")
    frame.to_hdf(os.path.join(run_folder, f"{event}_{recording_site}_{basename}.h5"), key="df", mode="w")


@pytest.fixture
def run_folder(tmp_path, base_input_parameters):
    """An output directory with two events on one recording site."""
    folder = tmp_path / "session_output_1"
    folder.mkdir()
    write_stores_list(
        str(folder),
        ["Dv1A", "Dv2A", "LNRW", "LNnR"],
        ["control_dms", "signal_dms", "rewarded", "unrewarded"],
    )
    for event in ("rewarded", "unrewarded"):
        write_psth(str(folder), event, "dms", "z_score_dms")
    return str(folder)


@pytest.fixture
def significance_parameters(base_input_parameters):
    base_input_parameters["computePsthSignificance"] = True
    base_input_parameters["selectForComputePsth"] = "z_score"
    base_input_parameters["psthComparisonsA"] = [""] * 10
    base_input_parameters["psthComparisonsB"] = [""] * 10
    return base_input_parameters


class TestPsthMetrics:
    @pytest.mark.parametrize(
        "selection, expected",
        [("z_score", ["z_score"]), ("dff", ["dff"]), ("Both", ["z_score", "dff"])],
    )
    def test_maps_the_selection_to_metrics(self, base_input_parameters, selection, expected):
        base_input_parameters["selectForComputePsth"] = selection

        assert psth_metrics(base_input_parameters) == expected


class TestComparisonName:
    def test_one_sample_names_the_single_event(self):
        name = comparison_name(event_a="rewarded", event_b=None, recording_site="dms", basename="z_score_dms")

        assert name == "rewarded_dms_z_score_dms"

    def test_two_sample_joins_both_events(self):
        name = comparison_name(event_a="rewarded", event_b="unrewarded", recording_site="dms", basename="z_score_dms")

        assert name == "rewarded_vs_unrewarded_dms_z_score_dms"


class TestReadPsthSamples:
    def test_returns_trials_transposed_with_the_time_axis(self, run_folder):
        samples, timestamps = read_psth_samples(
            filepath=run_folder, event="rewarded", recording_site="dms", basename="z_score_dms"
        )

        # Five trials of six timepoints, trial i filled with the value i.
        assert samples.shape == (5, 6)
        np.testing.assert_allclose(samples[:, 0], [0.0, 1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(timestamps, np.linspace(-1, 1, 6))

    def test_excludes_summary_and_binned_columns(self, tmp_path):
        folder = tmp_path / "binned_output_1"
        folder.mkdir()
        write_psth(str(folder), "rewarded", "dms", "z_score_dms", bins=True)

        samples, _ = read_psth_samples(
            filepath=str(folder), event="rewarded", recording_site="dms", basename="z_score_dms"
        )

        # The two bin columns sit between the trials and timestamps, so a positional
        # slice would pick them up as trials.
        assert samples.shape == (5, 6)

    def test_returns_none_when_the_psth_is_absent(self, run_folder):
        assert (
            read_psth_samples(filepath=run_folder, event="never_happened", recording_site="dms", basename="z_score_dms")
            is None
        )


class TestPlanComparisons:
    def test_tests_every_event_against_zero(self, run_folder, significance_parameters):
        planned = plan_comparisons(run_folder, significance_parameters)

        assert ("rewarded", None, "dms", "z_score_dms") in planned
        assert ("unrewarded", None, "dms", "z_score_dms") in planned

    def test_adds_only_the_named_pairs(self, run_folder, significance_parameters):
        significance_parameters["psthComparisonsA"] = ["rewarded"] + [""] * 9
        significance_parameters["psthComparisonsB"] = ["unrewarded"] + [""] * 9

        planned = plan_comparisons(run_folder, significance_parameters)

        pairs = [entry for entry in planned if entry[1] is not None]
        assert pairs == [("rewarded", "unrewarded", "dms", "z_score_dms")]

    def test_makes_no_pairs_when_none_are_named(self, run_folder, significance_parameters):
        planned = plan_comparisons(run_folder, significance_parameters)

        assert [entry for entry in planned if entry[1] is not None] == []

    def test_resolves_each_pair_within_every_site_and_metric(self, tmp_path, significance_parameters):
        folder = tmp_path / "two_site_output_1"
        folder.mkdir()
        write_stores_list(
            str(folder),
            ["Dv1A", "Dv2A", "Dv3A", "Dv4A", "LNRW", "LNnR"],
            ["control_dms", "signal_dms", "control_dls", "signal_dls", "rewarded", "unrewarded"],
        )
        for site in ("dms", "dls"):
            for metric in ("z_score", "dff"):
                for event in ("rewarded", "unrewarded"):
                    write_psth(str(folder), event, site, f"{metric}_{site}")
        significance_parameters["selectForComputePsth"] = "Both"
        significance_parameters["psthComparisonsA"] = ["rewarded"] + [""] * 9
        significance_parameters["psthComparisonsB"] = ["unrewarded"] + [""] * 9

        planned = plan_comparisons(str(folder), significance_parameters)

        pairs = sorted(entry for entry in planned if entry[1] is not None)
        assert pairs == [
            ("rewarded", "unrewarded", "dls", "dff_dls"),
            ("rewarded", "unrewarded", "dls", "z_score_dls"),
            ("rewarded", "unrewarded", "dms", "dff_dms"),
            ("rewarded", "unrewarded", "dms", "z_score_dms"),
        ]
        # Never a pair spanning two sites or two metrics: the basename always matches the site.
        for _, _, site, basename in pairs:
            assert basename.endswith(site)

    def test_skips_continuous_labels(self, run_folder, significance_parameters):
        planned = plan_comparisons(run_folder, significance_parameters)

        assert all(event not in ("control_dms", "signal_dms") for event, _, _, _ in planned)

    def test_rejects_a_comparison_naming_an_absent_event(self, run_folder, significance_parameters):
        significance_parameters["psthComparisonsA"] = ["rewarded"] + [""] * 9
        significance_parameters["psthComparisonsB"] = ["typo_event"] + [""] * 9

        with pytest.raises(ValueError, match="typo_event"):
            plan_comparisons(run_folder, significance_parameters)


class TestComputeComparison:
    def test_one_sample_reports_the_mean_and_its_sample_count(self):
        samples = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])

        significance = compute_comparison(
            samples_a=samples,
            samples_b=None,
            timestamps=np.array([0.0, 1.0]),
            minimum_consecutive_samples=1,
            significance_level=0.05,
            rng=np.random.default_rng(0),
        )

        assert list(significance.columns) == [
            "timestamps",
            "estimate",
            "ci_lower",
            "ci_upper",
            "significant",
            "alpha",
            "n",
        ]
        np.testing.assert_allclose(significance["estimate"].to_numpy(), [3.0, 3.0])
        assert significance["n"].tolist() == [5, 5]
        assert significance["alpha"].tolist() == [0.05, 0.05]

    def test_two_sample_reports_the_difference_and_both_counts(self):
        samples_a = np.array([[10.0], [11.0], [12.0], [13.0]])
        samples_b = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])

        significance = compute_comparison(
            samples_a=samples_a,
            samples_b=samples_b,
            timestamps=np.array([0.0]),
            minimum_consecutive_samples=1,
            significance_level=0.05,
            rng=np.random.default_rng(0),
        )

        assert list(significance.columns)[-3:] == ["alpha", "n", "n_b"]
        # Means are 11.5 and 2.0, so the difference is 9.5.
        np.testing.assert_allclose(significance["estimate"].to_numpy(), [9.5])
        assert significance["n"].tolist() == [4]
        assert significance["n_b"].tolist() == [5]

    def test_significance_respects_the_run_length_threshold(self):
        # A strong, entirely positive response over four timepoints. With a threshold of
        # four nothing survives; with three, all four timepoints do.
        samples = np.tile(np.array([[9.0], [10.0], [11.0], [12.0], [13.0]]), (1, 4))

        strict = compute_comparison(
            samples_a=samples,
            samples_b=None,
            timestamps=np.arange(4.0),
            minimum_consecutive_samples=4,
            significance_level=0.05,
            rng=np.random.default_rng(0),
        )
        lenient = compute_comparison(
            samples_a=samples,
            samples_b=None,
            timestamps=np.arange(4.0),
            minimum_consecutive_samples=3,
            significance_level=0.05,
            rng=np.random.default_rng(0),
        )

        assert strict["significant"].sum() == 0
        assert lenient["significant"].sum() == 4
