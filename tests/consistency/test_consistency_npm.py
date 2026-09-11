import shutil
from pathlib import Path

import numpy as np
import pytest

from guppy.testing import compare_output_folders
from guppy.testing.api import step1, step2, step3, step4
from guppy.utils.stores_list import read_stores_list, write_stores_list
from guppy_test_data import TESTING_DATA, event_ts_offset_for, recording_start_for

# Store ids the v1.3.0 reference outputs were generated under, mapped to the ones the
# current code derives. Step 2 names each raw store file after its store id and
# storesList.csv records those ids, so naming NPM's channels after their excitation
# wavelength (issue #336) moves files the frozen reference still holds under the old
# names. Every other reference file is keyed by store label, which is unchanged.
CONSISTENCY_CASES = [
    (
        "SampleData_Neurophotometrics/sampleData_NPM_2",
        "StandardOutputs_Neurophotometrics/sampleData_NPM_2/sampleData_NPM_2_output_1",
        {
            "file0_415nm_column6": "control_region",
            "file1_470nm_column6": "signal_region",
        },
        {"file0_chev6": "file0_415nm_column6", "file1_chev6": "file1_470nm_column6"},
        {"npm_split_events": None},
    ),
    (
        "SampleData_Neurophotometrics/sampleData_NPM_3",
        "StandardOutputs_Neurophotometrics/sampleData_NPM_3/sampleData_NPM_3_output_1",
        {
            "file0_415nm_column3": "control_region3",
            "file0_470nm_column3": "signal_region3",
            "event3": "ttl_region3",
        },
        {"file0_chev3": "file0_415nm_column3", "file0_chod3": "file0_470nm_column3"},
        {
            "npm_timestamp_column_name": "ComputerTimestamp",
            "npm_time_unit": "milliseconds",
            "npm_split_events": [False, True],
        },
    ),
    (
        "SampleData_Neurophotometrics/sampleData_NPM_4",
        "StandardOutputs_Neurophotometrics/sampleData_NPM_4/sampleData_NPM_4_output_1",
        {
            "file0_415nm_column1": "control_region1",
            "file0_470nm_column1": "signal_region1",
            "eventTrue": "ttl_true_region1",
        },
        {"file0_chev1": "file0_415nm_column1", "file0_chod1": "file0_470nm_column1"},
        {"npm_split_events": [False, True]},
    ),
    (
        "SampleData_Neurophotometrics/sampleData_NPM_5",
        "StandardOutputs_Neurophotometrics/sampleData_NPM_5/sampleData_NPM_5_output_1",
        {
            "file0_chev1": "control_region1",
            "file0_chod1": "signal_region1",
            "event0": "ttl_region1",
        },
        # Header-less: nothing names the LED that lit each frame, so the channels keep
        # the positional names the reference was generated under.
        {},
        # Its clock is in milliseconds, which only the user can state.
        {"npm_time_unit": "milliseconds", "npm_split_events": None},
    ),
]


def _reconcile_reference_store_ids(
    *, standard_output_dir: Path, destination: Path, reference_store_id_to_store_id: dict[str, str]
) -> Path:
    """Copy the reference output folder, renaming its raw per-store files to the current store ids.

    Parameters
    ----------
    standard_output_dir : Path
        The v1.3.0 reference output folder, which is left untouched.
    destination : Path
        Directory to write the reconciled copy into.
    reference_store_id_to_store_id : dict
        Maps each store id the reference was generated under to the one the current
        code derives. Empty when a session's store ids are unchanged.

    Returns
    -------
    Path
        The folder to compare against: the reconciled copy, or ``standard_output_dir``
        itself when the session's store ids are unchanged.
    """
    if not reference_store_id_to_store_id:
        return standard_output_dir

    shutil.copytree(standard_output_dir, destination)
    for reference_store_id, store_id in reference_store_id_to_store_id.items():
        (destination / f"{reference_store_id}.hdf5").rename(destination / f"{store_id}.hdf5")

    reference_store_array = read_stores_list(run_folder=destination)
    # Rebuilt rather than assigned into: the array read back is fixed-width, and the
    # current store ids are longer than the ones it was sized for.
    store_array = np.array(
        [
            [reference_store_id_to_store_id.get(store_id, store_id) for store_id in reference_store_array[0]],
            list(reference_store_array[1]),
        ],
        dtype=str,
    )
    write_stores_list(run_folder=destination, store_array=store_array)
    return destination


@pytest.mark.parametrize(
    "session_subdir, standard_output_subdir, store_id_to_store_label, reference_store_id_to_store_id, extra_kwargs",
    CONSISTENCY_CASES,
    ids=[
        "sample_npm_2",
        "sample_npm_3",
        "sample_npm_4",
        "sample_npm_5",
    ],
)
@pytest.mark.full_data
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_consistency(
    tmp_path,
    session_subdir,
    standard_output_subdir,
    store_id_to_store_label,
    reference_store_id_to_store_id,
    extra_kwargs,
):
    """
    Consistency test: run the full pipeline (Steps 2-5) and assert that the output
    is numerically identical (within tolerance) to the reference output from v1.3.0.

    NPM now demultiplexes in memory and writes no intermediate CSVs into the session
    folder, so correctness is validated end-to-end via the final HDF5/output comparison.
    """
    src_session = TESTING_DATA / session_subdir
    assert src_session.is_dir(), f"Sample data not found: {src_session}"

    standard_output_dir = TESTING_DATA / standard_output_subdir
    assert standard_output_dir.is_dir(), f"Standard output not found: {standard_output_dir}"

    tmp_base = tmp_path / "data_root"
    tmp_base.mkdir(parents=True, exist_ok=True)
    dest_name = src_session.name
    session_copy = tmp_base / dest_name
    shutil.copytree(src_session, session_copy)

    for d in list(Path(session_copy).glob(f"{dest_name}_output_*")):
        shutil.rmtree(d)
    params_fp = session_copy / "GuPPyParamtersUsed.json"
    if params_fp.exists():
        params_fp.unlink()

    common_kwargs = dict(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
    )

    selected_runs = {folder: ["1"] for folder in common_kwargs["selected_folders"]}
    step1(**common_kwargs, store_id_to_store_label=store_id_to_store_label, **extra_kwargs)
    step2(**common_kwargs, selected_runs=selected_runs, **extra_kwargs)
    step3(**common_kwargs, control_fit_method="OLS", selected_runs=selected_runs, **extra_kwargs)
    step4(**common_kwargs, selected_runs=selected_runs, **extra_kwargs)

    run_folders = sorted(list(Path(session_copy).glob(f"{dest_name}_output_*")))
    assert run_folders, f"No output directory found under {session_copy}"
    actual_output_dir = run_folders[0]

    expected_output_dir = _reconcile_reference_store_ids(
        standard_output_dir=standard_output_dir,
        destination=tmp_path / "expected_output",
        reference_store_id_to_store_id=reference_store_id_to_store_id,
    )

    compare_output_folders(
        actual_dir=actual_output_dir,
        expected_dir=str(expected_output_dir),
        event_ts_offset=event_ts_offset_for(tmp_base),
        # NPM now emits the acquisition clock (issue #407); the v1.3.0 reference was
        # generated with it re-zeroed, so its continuous timestamps sit one recording
        # start lower. The warm-up trim is measured from that same start, so the set of
        # retained samples — and every value derived from them — is unchanged.
        continuous_ts_offset=recording_start_for(tmp_base),
    )
