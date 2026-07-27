from guppy.orchestration import preprocess_view


def test_build_page_resolves_folders_and_composes(monkeypatch):
    monkeypatch.setattr(preprocess_view, "resolve_run_folders", lambda session_folders, params: ["/a/output_1"])
    calls = []
    monkeypatch.setattr(
        preprocess_view,
        "build_preprocess_view_page",
        lambda run_folders, remove_artifacts, plot_zScore_dff: calls.append(
            (run_folders, remove_artifacts, plot_zScore_dff)
        ),
    )

    preprocess_view._build_page(["/a"], {"removeArtifacts": True, "plot_zScore_dff": "z_score", "combine_data": False})

    assert calls == [(["/a/output_1"], True, "z_score")]


def test_exposes_route_factory_and_open_helper():
    assert callable(preprocess_view.build_preprocess_view)
    assert callable(preprocess_view.open_preprocess_view)
