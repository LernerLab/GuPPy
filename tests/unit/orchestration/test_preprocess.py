import numpy as np

from guppy.orchestration.preprocess import (
    execute_preprocessing_visualization,
    execute_zscore,
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
