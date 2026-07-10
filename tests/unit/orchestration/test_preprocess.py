import numpy as np
import pytest

from guppy.orchestration.preprocess import (
    execute_artifact_removal,
    execute_combine_data,
    execute_preprocessing_visualization,
    execute_zscore,
    extractTsAndSignal,
    visualize_artifact_removal,
    visualize_z_score,
    visualizeControlAndSignal,
)


class FakeArtifactRemovalWidget:
    def __init__(self, filepath, timestamps, control, signal, control_fit, plot_name, remove_artifacts):
        self.filepath = filepath
        self.timestamps = timestamps
        self.control = control
        self.signal = signal
        self.control_fit = control_fit
        self.plot_name = plot_name
        self.remove_artifacts = remove_artifacts


def test_execute_preprocessing_visualization_calls_visualizer_for_each_file(monkeypatch):
    filepath = "/tmp/session_output_1"
    discovered_paths = [
        f"{filepath}/z_score_DMS.hdf5",
        f"{filepath}/z_score_DLS.hdf5",
    ]

    monkeypatch.setattr("guppy.orchestration.preprocess.glob.glob", lambda pattern: discovered_paths)

    read_calls = []

    def fake_read_hdf5(name, path, dataset_name):
        read_calls.append((name, path, dataset_name))
        return np.array([0.0, 1.0, 2.0])

    visualizer_calls = []

    def fake_visualize_preprocessing(*, suptitle, title, x, y):
        visualizer_calls.append((suptitle, title, x.copy(), y.copy()))
        return "figure", "axes"

    monkeypatch.setattr("guppy.orchestration.preprocess.read_hdf5", fake_read_hdf5)
    monkeypatch.setattr("guppy.orchestration.preprocess.visualize_preprocessing", fake_visualize_preprocessing)

    execute_preprocessing_visualization(filepath, visualization_type="z_score")

    assert len(visualizer_calls) == 2
    assert visualizer_calls[0][0] == "session_output_1"
    assert visualizer_calls[0][1] == "z_score_DLS"
    assert visualizer_calls[1][1] == "z_score_DMS"
    assert read_calls[0] == ("timeCorrection_DLS", filepath, "timestampNew")
    assert read_calls[1] == ("", f"{filepath}/z_score_DLS.hdf5", "data")


def test_visualize_control_and_signal_returns_widgets_without_real_panel(monkeypatch):
    filepath = "/tmp/session_output_1"
    control_path = f"{filepath}/control_DMS.hdf5"
    signal_path = f"{filepath}/signal_DMS.hdf5"

    monkeypatch.setattr(
        "guppy.orchestration.preprocess.find_files",
        lambda path, pattern, ignore_case=True: [control_path] if "control" in pattern else [signal_path],
    )

    def fake_read_hdf5(name, path, dataset_name):
        if dataset_name == "timestampNew":
            return np.array([0.0, 1.0, 2.0])
        if "cntrl_sig_fit" in path:
            return np.array([0.5, 0.5, 0.5])
        if "control" in path:
            return np.array([1.0, 1.0, 1.0])
        return np.array([2.0, 2.0, 2.0])

    monkeypatch.setattr("guppy.orchestration.preprocess.read_hdf5", fake_read_hdf5)
    monkeypatch.setattr("guppy.orchestration.preprocess.ArtifactRemovalWidget", FakeArtifactRemovalWidget)

    widgets = visualizeControlAndSignal(filepath, removeArtifacts=False)

    assert len(widgets) == 1
    assert widgets[0].filepath == filepath
    assert widgets[0].remove_artifacts is False
    assert widgets[0].plot_name == ["control_DMS", "signal_DMS", "cntrl_sig_fit_DMS"]


def test_visualize_z_score_executes_both_plots_and_show(monkeypatch, base_input_parameters):
    folder_names = [["/tmp/session_output_1"]]

    base_input_parameters["combine_data"] = True
    base_input_parameters["removeArtifacts"] = False
    base_input_parameters["plot_zScore_dff"] = "Both"

    control_signal_calls = []
    visualize_calls = []
    show_calls = []

    monkeypatch.setattr(
        "guppy.orchestration.preprocess.visualizeControlAndSignal",
        lambda filepath, removeArtifacts: control_signal_calls.append((filepath, removeArtifacts)) or [object()],
    )
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.execute_preprocessing_visualization",
        lambda filepath, visualization_type: visualize_calls.append((filepath, visualization_type)),
    )
    monkeypatch.setattr("guppy.orchestration.preprocess.plt.show", lambda: show_calls.append(True))

    visualize_z_score(base_input_parameters, folder_names)

    assert control_signal_calls == [("/tmp/session_output_1", False)]
    assert visualize_calls == [
        ("/tmp/session_output_1", "z_score"),
        ("/tmp/session_output_1", "dff"),
    ]
    assert len(show_calls) == 1


def test_visualize_artifact_removal_invokes_widget_visualization_and_show(monkeypatch, base_input_parameters):
    folder_names = [["/tmp/session_output_1"], ["/tmp/session_output_2"]]

    base_input_parameters["combine_data"] = True

    widget_calls = []
    show_calls = []

    monkeypatch.setattr(
        "guppy.orchestration.preprocess.visualizeControlAndSignal",
        lambda filepath, removeArtifacts: widget_calls.append((filepath, removeArtifacts)),
    )
    monkeypatch.setattr("guppy.orchestration.preprocess.plt.show", lambda: show_calls.append(True))

    visualize_artifact_removal(folder_names, base_input_parameters)

    assert widget_calls == [
        ("/tmp/session_output_1", True),
        ("/tmp/session_output_2", True),
    ]
    assert len(show_calls) == 1


# ── error paths ───────────────────────────────────────────────────────────────


def test_visualize_control_and_signal_raises_for_unequal_file_counts(monkeypatch):
    """Three control files but only two signal files → odd total → raises with counts."""
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.find_files",
        lambda path, pattern, ignore_case=True: (
            ["/tmp/ctrl_a.hdf5", "/tmp/ctrl_b.hdf5", "/tmp/ctrl_c.hdf5"]
            if "control" in pattern
            else ["/tmp/sig_a.hdf5", "/tmp/sig_b.hdf5"]
        ),
    )
    with pytest.raises(ValueError, match="Unequal number of control and signal files"):
        visualizeControlAndSignal("/tmp/session_output_1", removeArtifacts=False)


def test_execute_zscore_raises_for_pair_name_mismatch(monkeypatch, base_input_parameters):
    """control_dms paired with signal_vms (different suffixes) — raises with both suffixes."""
    folder_names = [["/tmp/session_output_1"]]
    base_input_parameters["combine_data"] = True

    monkeypatch.setattr(
        "guppy.orchestration.preprocess.find_files",
        lambda path, pattern, ignore_case=True: (
            ["/tmp/session_output_1/control_dms.hdf5"]
            if "control" in pattern
            else ["/tmp/session_output_1/signal_vms.hdf5"]
        ),
    )

    with pytest.raises(ValueError) as exception_info:
        execute_zscore(folder_names, base_input_parameters)
    message = str(exception_info.value)
    assert "Pair name mismatch" in message
    assert "dms" in message
    assert "vms" in message


def test_execute_combine_data_raises_for_mismatched_sampling_rates(monkeypatch, base_input_parameters):
    """When timeCorrection_*.hdf5 files report different sampling rates, the message
    lists both rates and the offending paths."""
    folder_names = ["/tmp/session_a", "/tmp/session_b"]
    store_array = np.array([["ctrl0", "sig0"], ["control_dms", "signal_dms"]])

    monkeypatch.setattr(
        "guppy.orchestration.preprocess.select_run_folders",
        lambda session, selected: (
            [folder_names[0] + "/run_folder"] if "session_a" in session else [folder_names[1] + "/run_folder"]
        ),
    )
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.glob.glob",
        lambda pattern: (
            [f"{folder_names[0]}/run_folder/timeCorrection_dms.hdf5"]
            if "session_a" in pattern
            else [f"{folder_names[1]}/run_folder/timeCorrection_dms.hdf5"] if "session_b" in pattern else []
        ),
    )

    rates = iter([np.array([100.0]), np.array([250.0])])
    monkeypatch.setattr("guppy.orchestration.preprocess.read_hdf5", lambda *a, **k: next(rates))
    monkeypatch.setattr("guppy.orchestration.preprocess.np.genfromtxt", lambda *a, **k: store_array)

    with pytest.raises(ValueError, match="sampling rates differ"):
        execute_combine_data(folder_names, base_input_parameters, store_array)


def test_execute_zscore_shows_plot_when_not_headless(monkeypatch, base_input_parameters):
    folder_names = [["/tmp/session_output_1"]]

    base_input_parameters["combine_data"] = True

    monkeypatch.delenv("GUPPY_BASE_DIR", raising=False)

    monkeypatch.setattr(
        "guppy.orchestration.preprocess.find_files",
        lambda filepath, pattern, ignore_case=True: [f"{filepath}/{pattern.replace('*', 'DMS')}.hdf5"],
    )
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.read_corrected_data",
        lambda control_path, signal_path, filepath, name: (
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 2.0, 2.0]),
            np.array([0.0, 1.0, 2.0]),
        ),
    )
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.get_coords", lambda filepath, name, ts_new, remove: np.array([])
    )
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.compute_z_score",
        lambda *args, **kwargs: (
            np.array([0.0, 0.1, 0.2]),
            np.array([0.0, 0.2, 0.4]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        ),
    )

    write_calls = []
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.write_zscore",
        lambda filepath, name, z_score, dff, control_fit, temporary_control_array: write_calls.append((filepath, name)),
    )
    monkeypatch.setattr("guppy.orchestration.preprocess.writeToFile", lambda text, file_path: None)

    show_calls = []
    monkeypatch.setattr("guppy.orchestration.preprocess.plt.show", lambda: show_calls.append(True))

    execute_zscore(folder_names, base_input_parameters)

    assert write_calls == [("/tmp/session_output_1", "DMS")]
    assert len(show_calls) == 1


@pytest.fixture
def stub_artifact_removal_io(monkeypatch):
    """Patch execute_artifact_removal's disk I/O so only its visualization branch is exercised."""
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.select_run_folders", lambda session, selected: ["/tmp/session_1/run_1"]
    )
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.np.genfromtxt", lambda *a, **k: np.array([["ctrl0"], ["control_dms"]])
    )
    monkeypatch.setattr("guppy.orchestration.preprocess.read_corrected_data_dict", lambda filepath, store_array: {})
    monkeypatch.setattr("guppy.orchestration.preprocess.read_corrected_timestamps_pairwise", lambda filepath: ({}, {}))
    monkeypatch.setattr("guppy.orchestration.preprocess.read_coords_pairwise", lambda filepath, tsNew: {})
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.read_corrected_ttl_timestamps", lambda filepath, store_array: {}
    )
    monkeypatch.setattr("guppy.orchestration.preprocess.remove_artifacts", lambda *a, **k: ({}, {}, {}))
    monkeypatch.setattr("guppy.orchestration.preprocess.write_artifact_removal", lambda *a, **k: None)
    monkeypatch.setattr("guppy.orchestration.preprocess.writeToFile", lambda text, file_path: None)


def test_execute_artifact_removal_visualizes_when_not_headless(
    monkeypatch, base_input_parameters, stub_artifact_removal_io
):
    """When not headless, execute_artifact_removal calls visualize_artifact_removal after processing."""
    base_input_parameters["combine_data"] = False
    monkeypatch.delenv("GUPPY_BASE_DIR", raising=False)

    visualize_calls = []
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.visualize_artifact_removal",
        lambda folders, params: visualize_calls.append((folders, params)),
    )

    execute_artifact_removal(["/tmp/session_1"], base_input_parameters)

    assert visualize_calls == [(["/tmp/session_1"], base_input_parameters)]


def test_execute_artifact_removal_skips_visualization_when_headless(
    monkeypatch, base_input_parameters, stub_artifact_removal_io
):
    """When headless, execute_artifact_removal does not open the artifact-removal visualization."""
    base_input_parameters["combine_data"] = False
    monkeypatch.setenv("GUPPY_BASE_DIR", "/tmp/base")

    visualize_calls = []
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.visualize_artifact_removal",
        lambda folders, params: visualize_calls.append((folders, params)),
    )

    execute_artifact_removal(["/tmp/session_1"], base_input_parameters)

    assert visualize_calls == []


@pytest.fixture
def stub_extract_ts_and_signal_io(monkeypatch):
    """Patch extractTsAndSignal's sub-steps so only its dispatch/visualization branches run."""
    monkeypatch.setattr("guppy.orchestration.preprocess.save_parameters", lambda *, inputParameters: None)
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.select_run_folders", lambda session, selected: ["/tmp/session_1/run_1"]
    )
    monkeypatch.setattr("guppy.orchestration.preprocess.writeToFile", lambda text, file_path: None)
    monkeypatch.setattr("guppy.orchestration.preprocess.execute_timestamp_correction", lambda folders, params: None)
    monkeypatch.setattr("guppy.orchestration.preprocess.execute_zscore", lambda folders, params: None)
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.check_storeslistfile", lambda folders: np.array([["ctrl0"], ["control_dms"]])
    )
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.execute_combine_data", lambda folders, params, store_array: ["/tmp/combined_1"]
    )
    monkeypatch.setattr("guppy.orchestration.preprocess.write_combined_stores_list", lambda folders, store_array: None)


def test_extract_ts_and_signal_visualizes_and_removes_artifacts_when_not_headless(
    monkeypatch, base_input_parameters, stub_extract_ts_and_signal_io
):
    """Non-combine path: extractTsAndSignal visualizes z-score and runs artifact removal when not headless."""
    base_input_parameters["session_folders"] = ["/tmp/session_1"]
    base_input_parameters["combine_data"] = False
    base_input_parameters["removeArtifacts"] = True
    monkeypatch.delenv("GUPPY_BASE_DIR", raising=False)

    visualize_calls = []
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.visualize_z_score",
        lambda params, folders: visualize_calls.append((params, folders)),
    )
    artifact_calls = []
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.execute_artifact_removal",
        lambda folders, params: artifact_calls.append((folders, params)),
    )

    extractTsAndSignal(base_input_parameters)

    assert visualize_calls == [(base_input_parameters, ["/tmp/session_1"])]
    assert artifact_calls == [(["/tmp/session_1"], base_input_parameters)]


def test_extract_ts_and_signal_combine_visualizes_when_not_headless(
    monkeypatch, base_input_parameters, stub_extract_ts_and_signal_io
):
    """Combine path: extractTsAndSignal visualizes z-score and runs artifact removal on the combined folders."""
    base_input_parameters["session_folders"] = ["/tmp/session_1"]
    base_input_parameters["combine_data"] = True
    base_input_parameters["removeArtifacts"] = True
    monkeypatch.delenv("GUPPY_BASE_DIR", raising=False)

    visualize_calls = []
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.visualize_z_score",
        lambda params, folders: visualize_calls.append((params, folders)),
    )
    artifact_calls = []
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.execute_artifact_removal",
        lambda folders, params: artifact_calls.append((folders, params)),
    )

    extractTsAndSignal(base_input_parameters)

    assert visualize_calls == [(base_input_parameters, ["/tmp/combined_1"])]
    assert artifact_calls == [(["/tmp/combined_1"], base_input_parameters)]


def test_extract_ts_and_signal_combine_skips_visualization_when_headless(
    monkeypatch, base_input_parameters, stub_extract_ts_and_signal_io
):
    """Combine path when headless: z-score visualization is skipped but artifact removal still runs."""
    base_input_parameters["session_folders"] = ["/tmp/session_1"]
    base_input_parameters["combine_data"] = True
    base_input_parameters["removeArtifacts"] = True
    monkeypatch.setenv("GUPPY_BASE_DIR", "/tmp/base")

    visualize_calls = []
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.visualize_z_score",
        lambda params, folders: visualize_calls.append((params, folders)),
    )
    artifact_calls = []
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.execute_artifact_removal",
        lambda folders, params: artifact_calls.append((folders, params)),
    )

    extractTsAndSignal(base_input_parameters)

    assert visualize_calls == []
    assert artifact_calls == [(["/tmp/combined_1"], base_input_parameters)]
