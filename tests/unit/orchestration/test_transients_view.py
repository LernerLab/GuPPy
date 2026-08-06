from guppy.orchestration import transients_view


def test_build_page_resolves_folders_and_composes(monkeypatch):
    monkeypatch.setattr(transients_view, "resolve_run_folders", lambda session_folders, params: ["/a/output_1"])
    calls = []
    monkeypatch.setattr(
        transients_view,
        "build_peaks_view_page",
        lambda run_folders, select_for_transients: calls.append((run_folders, select_for_transients)),
    )

    transients_view._build_page(["/a"], {"selectForTransientsComputation": "z_score", "combine_data": False})

    assert calls == [(["/a/output_1"], "z_score")]


def test_exposes_route_factory_and_open_helper():
    assert callable(transients_view.build_transients_view)
    assert callable(transients_view.open_transients_view)
