import os
import shutil

import numpy as np
import pandas as pd
import pytest

from guppy.analysis.io_utils import PSTH_SIGNIFICANCE_DIRNAME
from guppy.analysis.standard_io import read_psth_significance_from_hdf5
from guppy.testing.api import group_analysis, label_groups, step1, step2, step3, step4
from guppy.utils.utils import parse_run_name
from guppy_test_data import STUBBED_TESTING_DATA

from .integration_helpers import _locate_output_directory

# Trial counts in the TDT stub: port_entries fires 4 times, unrewarded_nose_pokes 18,
# and rewarded_nose_pokes only twice. Labeling all three exercises the two comparison
# kinds and the too-few-trials skip from a single session.
STORE_ID_TO_STORE_LABEL = {
    "Dv1A": "control_dms",
    "Dv2A": "signal_dms",
    "PrtN": "port_entries",
    "LNnR": "unrewarded_nose_pokes",
    "LNRW": "rewarded_nose_pokes",
}
SESSION_SUBDIR = "tdt/Photo_63_207-181030-103332"
COMPARISON = ("port_entries", "unrewarded_nose_pokes")
UNDERPOWERED_EVENT = "rewarded_nose_pokes"

EXPECTED_ONE_SAMPLE_COLUMNS = ["timestamps", "estimate", "ci_lower", "ci_upper", "significant", "n"]
EXPECTED_TWO_SAMPLE_COLUMNS = EXPECTED_ONE_SAMPLE_COLUMNS + ["n_b"]


def results_path(output_directory):
    return os.path.join(output_directory, PSTH_SIGNIFICANCE_DIRNAME)


@pytest.fixture(scope="module")
def preprocessed_session(tmp_path_factory):
    source_session = os.path.join(str(STUBBED_TESTING_DATA), SESSION_SUBDIR)
    base_directory = tmp_path_factory.mktemp("integration_psth_significance")
    session = base_directory / os.path.basename(source_session)
    # Output directories are gitignored, so a stale one from a previous local run
    # would otherwise seed this session.
    shutil.copytree(source_session, session, ignore=shutil.ignore_patterns("*_output_*"))

    step1(
        base_dir=str(base_directory),
        selected_folders=[str(session)],
        store_id_to_store_label=STORE_ID_TO_STORE_LABEL,
    )
    output_directory = _locate_output_directory(session_copy=str(session))
    selected_runs = {str(session): [parse_run_name(output_directory)]}

    step2(base_dir=str(base_directory), selected_folders=[str(session)], selected_runs=selected_runs)
    step3(base_dir=str(base_directory), selected_folders=[str(session)], selected_runs=selected_runs)

    return {
        "base_directory": str(base_directory),
        "session": str(session),
        "selected_runs": selected_runs,
        "output_directory": _locate_output_directory(session_copy=str(session)),
    }


@pytest.fixture(scope="module")
def significance_output(preprocessed_session):
    step4(
        base_dir=preprocessed_session["base_directory"],
        selected_folders=[preprocessed_session["session"]],
        selected_runs=preprocessed_session["selected_runs"],
        compute_psth_significance=True,
        psth_comparisons=[COMPARISON],
    )
    return preprocessed_session["output_directory"]


class TestSessionScopeSignificance:
    def test_writes_a_file_for_every_planned_comparison(self, significance_output):
        written = sorted(os.listdir(results_path(significance_output)))

        assert written == [
            "significance_port_entries_dms_z_score_dms.csv",
            "significance_port_entries_dms_z_score_dms.h5",
            "significance_port_entries_vs_unrewarded_nose_pokes_dms_z_score_dms.csv",
            "significance_port_entries_vs_unrewarded_nose_pokes_dms_z_score_dms.h5",
            "significance_unrewarded_nose_pokes_dms_z_score_dms.csv",
            "significance_unrewarded_nose_pokes_dms_z_score_dms.h5",
        ]

    def test_an_event_with_too_few_trials_is_skipped(self, significance_output):
        # rewarded_nose_pokes fires twice, below the three the bootstrap needs. It is
        # skipped rather than written with a meaningless interval.
        written = os.listdir(results_path(significance_output))

        assert not any(name.startswith("significance_" + UNDERPOWERED_EVENT) for name in written)

    def test_one_sample_schema(self, significance_output):
        significance = read_psth_significance_from_hdf5(
            filepath=results_path(significance_output), name="port_entries_dms_z_score_dms"
        )

        assert list(significance.columns) == EXPECTED_ONE_SAMPLE_COLUMNS

    def test_two_sample_schema(self, significance_output):
        significance = read_psth_significance_from_hdf5(
            filepath=results_path(significance_output),
            name="port_entries_vs_unrewarded_nose_pokes_dms_z_score_dms",
        )

        assert list(significance.columns) == EXPECTED_TWO_SAMPLE_COLUMNS

    def test_sample_counts_match_the_psth_trials(self, significance_output):
        psth = pd.read_hdf(os.path.join(significance_output, "port_entries_dms_z_score_dms.h5"), key="df", mode="r")
        num_trials = len([column for column in psth.columns if column not in ("timestamps", "mean", "err")])

        significance = read_psth_significance_from_hdf5(
            filepath=results_path(significance_output), name="port_entries_dms_z_score_dms"
        )

        assert significance["n"].iloc[0] == num_trials

    def test_time_axis_matches_the_psth(self, significance_output):
        psth = pd.read_hdf(os.path.join(significance_output, "port_entries_dms_z_score_dms.h5"), key="df", mode="r")
        significance = read_psth_significance_from_hdf5(
            filepath=results_path(significance_output), name="port_entries_dms_z_score_dms"
        )

        np.testing.assert_allclose(significance["timestamps"].to_numpy(), psth["timestamps"].to_numpy(), rtol=1e-6)

    def test_estimate_lies_inside_its_confidence_interval(self, significance_output):
        significance = read_psth_significance_from_hdf5(
            filepath=results_path(significance_output), name="port_entries_dms_z_score_dms"
        )

        # Timepoints where too few trials overlap have no computable interval; the
        # containment property is asserted over the rest.
        computable = np.isfinite(significance["ci_lower"]) & np.isfinite(significance["ci_upper"])
        assert computable.any()
        assert (significance["ci_lower"][computable] <= significance["estimate"][computable]).all()
        assert (significance["estimate"][computable] <= significance["ci_upper"][computable]).all()

    def test_an_uncomputable_timepoint_is_never_marked_significant(self, significance_output):
        # NaN padding at the window edges leaves some timepoints with too few overlapping
        # trials to bootstrap. They must fall out as not significant rather than either way.
        significance = read_psth_significance_from_hdf5(
            filepath=results_path(significance_output), name="port_entries_dms_z_score_dms"
        )

        uncomputable = ~np.isfinite(significance["ci_lower"])
        assert uncomputable.any()
        assert not significance["significant"][uncomputable].any()

    def test_significance_is_only_marked_where_the_interval_excludes_zero(self, significance_output):
        significance = read_psth_significance_from_hdf5(
            filepath=results_path(significance_output), name="port_entries_dms_z_score_dms"
        )

        marked = significance["significant"] == 1
        excludes_zero = (significance["ci_lower"] > 0) | (significance["ci_upper"] < 0)
        # The run-length filter only ever removes marks, so every mark implies exclusion.
        assert (excludes_zero[marked]).all()

    def test_csv_matches_the_hdf5(self, significance_output):
        significance = read_psth_significance_from_hdf5(
            filepath=results_path(significance_output), name="port_entries_dms_z_score_dms"
        )
        from_csv = pd.read_csv(
            os.path.join(results_path(significance_output), "significance_port_entries_dms_z_score_dms.csv")
        )

        pd.testing.assert_frame_equal(significance.reset_index(drop=True), from_csv, check_dtype=False)

    def test_parameters_are_recorded_in_the_snapshot(self, significance_output):
        import json

        with open(os.path.join(significance_output, "GuPPyParamtersUsed.json")) as parameters_file:
            saved = json.load(parameters_file)

        assert saved["computePsthSignificance"] is True
        assert saved["psthComparisonsA"] == [COMPARISON[0]]
        assert saved["psthComparisonsB"] == [COMPARISON[1]]


class TestSignificanceIsOptional:
    def test_step4_without_the_flag_writes_no_results(self, tmp_path_factory):
        source_session = os.path.join(str(STUBBED_TESTING_DATA), SESSION_SUBDIR)
        base_directory = tmp_path_factory.mktemp("integration_psth_significance_off")
        session = base_directory / os.path.basename(source_session)
        shutil.copytree(source_session, session, ignore=shutil.ignore_patterns("*_output_*"))

        step1(
            base_dir=str(base_directory),
            selected_folders=[str(session)],
            store_id_to_store_label=STORE_ID_TO_STORE_LABEL,
        )
        output_directory = _locate_output_directory(session_copy=str(session))
        selected_runs = {str(session): [parse_run_name(output_directory)]}
        step2(base_dir=str(base_directory), selected_folders=[str(session)], selected_runs=selected_runs)
        step3(base_dir=str(base_directory), selected_folders=[str(session)], selected_runs=selected_runs)
        step4(base_dir=str(base_directory), selected_folders=[str(session)], selected_runs=selected_runs)

        assert not os.path.exists(results_path(output_directory))


class TestGroupScopeSignificance:
    @pytest.fixture(scope="class")
    def group_folder(self, tmp_path_factory):
        """A three-member group built from repeats of the stubbed session."""
        source_session = os.path.join(str(STUBBED_TESTING_DATA), SESSION_SUBDIR)
        base_directory = tmp_path_factory.mktemp("integration_psth_significance_group")

        member_run_folders = []
        for member_index in range(3):
            session = base_directory / f"member_{member_index}"
            shutil.copytree(source_session, session, ignore=shutil.ignore_patterns("*_output_*"))
            step1(
                base_dir=str(base_directory),
                selected_folders=[str(session)],
                store_id_to_store_label=STORE_ID_TO_STORE_LABEL,
            )
            output_directory = _locate_output_directory(session_copy=str(session))
            selected_runs = {str(session): [parse_run_name(output_directory)]}
            step2(base_dir=str(base_directory), selected_folders=[str(session)], selected_runs=selected_runs)
            step3(base_dir=str(base_directory), selected_folders=[str(session)], selected_runs=selected_runs)
            step4(base_dir=str(base_directory), selected_folders=[str(session)], selected_runs=selected_runs)
            member_run_folders.append(output_directory)

        label_groups(
            member_run_folders=member_run_folders,
            destination_directory=str(base_directory),
            group_name="saline",
        )
        group_folder = os.path.join(str(base_directory), "saline_group")
        group_analysis(
            base_dir=str(base_directory),
            selected_group_folders=[group_folder],
            compute_psth_significance=True,
            psth_comparisons=[COMPARISON],
        )
        return group_folder, len(member_run_folders)

    def test_writes_results_into_the_group_directory(self, group_folder):
        folder, _ = group_folder

        assert os.path.exists(results_path(folder))
        assert "significance_port_entries_dms_z_score_dms.h5" in os.listdir(results_path(folder))

    def test_resamples_sessions_rather_than_trials(self, group_folder):
        folder, member_count = group_folder

        significance = read_psth_significance_from_hdf5(
            filepath=results_path(folder), name="port_entries_dms_z_score_dms"
        )

        # The recorded n is the group's member count, not any member's trial count --
        # the check that the resampling unit switched with the scope.
        assert significance["n"].iloc[0] == member_count

    def test_pairwise_comparison_records_both_member_counts(self, group_folder):
        folder, member_count = group_folder

        significance = read_psth_significance_from_hdf5(
            filepath=results_path(folder),
            name="port_entries_vs_unrewarded_nose_pokes_dms_z_score_dms",
        )

        assert significance["n"].iloc[0] == member_count
        assert significance["n_b"].iloc[0] == member_count
