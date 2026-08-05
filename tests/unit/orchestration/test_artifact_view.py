from guppy.orchestration import artifact_view


def test_build_page_resolves_folders_and_composes(monkeypatch):
    monkeypatch.setattr(artifact_view, "resolve_run_folders", lambda session_folders, params: ["/a/output_1"])
    calls = []
    monkeypatch.setattr(
        artifact_view,
        "build_artifact_review_page",
        lambda *, run_folders, plot_zScore_dff: calls.append((run_folders, plot_zScore_dff)),
    )

    artifact_view._build_page(["/a"], {"plot_zScore_dff": "z_score", "combine_data": False})

    assert calls == [(["/a/output_1"], "z_score")]


def test_exposes_route_factory_and_open_helper():
    assert callable(artifact_view.build_artifact_view)
    assert callable(artifact_view.open_artifact_view)
