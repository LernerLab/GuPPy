import glob
import os
import shutil

import pytest
from conftest import TESTING_DATA, event_ts_offset_for

from guppy.testing import compare_output_folders
from guppy.testing.api import step1, step2, step3, step4

CONSISTENCY_CASES = [
    (
        "SampleData_Clean/Photo_63_207-181030-103332",
        "StandardOutputs_Clean/Photo_63_207-181030-103332/Photo_63_207-181030-103332_output_1",
        {
            "Dv1A": "control_DMS",
            "Dv2A": "signal_DMS",
            "PrtN": "port_entries_dms",
        },
        {},
    ),
    (
        "SampleData_Clean/Photometry-161823",
        "StandardOutputs_Clean/Photometry-161823/Photometry-161823_output_1",
        # PAB/ is an epoc store that splits into one sub-event per marker value. These are now
        # enumerated at step 1, so the map references the split sub-events directly (labeled to
        # match the v1.3.0 reference names). Two artifacts the old flow produced are intentionally
        # gone and mapped to None (the untouched reference still contains them): the unsplit-parent
        # orphan PAB_.hdf5 that read() also emitted, and the .cache_storesList.csv left behind when
        # read() renamed/rewrote store_array — read() no longer mutates store_array at all.
        {
            "405R": "control_region",
            "490R": "signal_region",
            "PAB0": "PAB_0",
            "PAB16": "PAB_16",
            "PAB2064": "PAB_2064",
        },
        {"PAB_.hdf5": None, ".cache_storesList.csv": None},
    ),
]


@pytest.mark.parametrize(
    "session_subdir, standard_output_subdir, store_id_to_store_label, name_map",
    CONSISTENCY_CASES,
    ids=[
        "tdt_clean",
        "tdt_split_event",
    ],
)
@pytest.mark.full_data
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_consistency(
    tmp_path,
    session_subdir,
    standard_output_subdir,
    store_id_to_store_label,
    name_map,
):
    """
    Consistency test: run the full pipeline (Steps 2-5) and assert that the output
    is numerically identical (within tolerance) to the reference output from v1.3.0.
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

    for d in glob.glob(os.path.join(session_copy, f"{dest_name}_output_*")):
        shutil.rmtree(d)
    params_fp = session_copy / "GuPPyParamtersUsed.json"
    if params_fp.exists():
        params_fp.unlink()

    common_kwargs = dict(
        base_dir=str(tmp_base),
        selected_folders=[str(session_copy)],
    )

    selected_runs = {folder: ["1"] for folder in common_kwargs["selected_folders"]}
    step1(**common_kwargs, store_id_to_store_label=store_id_to_store_label)
    step2(**common_kwargs, selected_runs=selected_runs)
    step3(**common_kwargs, control_fit_method="OLS", selected_runs=selected_runs)
    step4(**common_kwargs, selected_runs=selected_runs)

    run_folders = sorted(glob.glob(os.path.join(session_copy, f"{dest_name}_output_*")))
    assert run_folders, f"No output directory found under {session_copy}"
    actual_output_dir = run_folders[0]

    compare_output_folders(
        actual_dir=actual_output_dir,
        expected_dir=str(standard_output_dir),
        event_ts_offset=event_ts_offset_for(tmp_base),
        name_map=name_map,
    )
