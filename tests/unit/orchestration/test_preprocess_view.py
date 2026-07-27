import panel as pn

from guppy.orchestration import preprocess_view
from guppy.orchestration.preprocess_view import (
    _resolve_run_folders,
    build_preprocess_view,
    open_preprocess_view,
)


class TestResolveRunFolders:
    def test_non_combine_returns_per_session_run_folders(self, monkeypatch):
        monkeypatch.setattr(preprocess_view, "select_run_folders", lambda session, runs: [session + "/output_1"])
        result = _resolve_run_folders(["/a", "/b"], {"combine_data": False, "selected_runs": {}})
        assert result == ["/a/output_1", "/b/output_1"]

    def test_combine_returns_first_folder_of_each_group(self, monkeypatch):
        monkeypatch.setattr(preprocess_view, "select_run_folders", lambda session, runs: [session + "/output_1"])
        monkeypatch.setattr(
            preprocess_view, "get_all_stores_for_combining_data", lambda folders: [[folders[0], folders[1]]]
        )
        result = _resolve_run_folders(["/a", "/b"], {"combine_data": True, "selected_runs": {}})
        assert result == ["/a/output_1"]


class TestOpenPreprocessView:
    def test_registers_token_and_opens_view_url(self, monkeypatch):
        preprocess_view._PENDING_VIEWS.clear()
        monkeypatch.setattr(preprocess_view, "_current_href", lambda: "http://localhost:5006/")
        opened = []
        monkeypatch.setattr(preprocess_view.webbrowser, "open", lambda url: opened.append(url))

        params = {"removeArtifacts": False, "plot_zScore_dff": "None", "combine_data": False}
        open_preprocess_view(["/a"], params)

        assert len(preprocess_view._PENDING_VIEWS) == 1
        token = next(iter(preprocess_view._PENDING_VIEWS))
        assert preprocess_view._PENDING_VIEWS[token] == (["/a"], params)
        assert opened == [f"http://localhost:5006/preprocess-view?token={token}"]


class TestBuildPreprocessView:
    def test_expired_token_shows_notice(self, panel_extension, monkeypatch):
        monkeypatch.setattr(preprocess_view, "_read_token", lambda: "bogus")
        preprocess_view._PENDING_VIEWS.pop("bogus", None)

        template = build_preprocess_view()

        text = " ".join(str(o.object) for o in template.main if hasattr(o, "object"))
        assert "expired" in text.lower()

    def test_valid_token_composes_page_from_resolved_folders(self, panel_extension, monkeypatch):
        monkeypatch.setattr(preprocess_view, "_read_token", lambda: "tok")
        preprocess_view._PENDING_VIEWS["tok"] = (
            ["/a"],
            {"removeArtifacts": True, "plot_zScore_dff": "z_score", "combine_data": False, "selected_runs": {}},
        )
        monkeypatch.setattr(preprocess_view, "_resolve_run_folders", lambda folders, params: ["/a/output_1"])
        compose_calls = []
        monkeypatch.setattr(
            preprocess_view,
            "build_preprocess_view_page",
            lambda run_folders, remove_artifacts, plot_zScore_dff: compose_calls.append(
                (run_folders, remove_artifacts, plot_zScore_dff)
            )
            or pn.pane.Markdown("page"),
        )
        # No live session in the test, so stub the session-destroyed registration.
        monkeypatch.setattr(preprocess_view.pn.state, "on_session_destroyed", lambda callback: None)

        build_preprocess_view()

        assert compose_calls == [(["/a/output_1"], True, "z_score")]
        preprocess_view._PENDING_VIEWS.pop("tok", None)
