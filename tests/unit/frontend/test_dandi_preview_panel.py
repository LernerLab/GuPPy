"""Unit tests for the panel that renders what one DANDI asset holds.

The streaming preview is a constructor injection point, so every test here runs against an
in-memory stand-in with no archive access.
"""

import holoviews as hv
import numpy as np
import pytest

from guppy.frontend.dandi_preview_panel import (
    CHANNEL_COLUMNS,
    DandiPreviewPanel,
    build_trace_overlay,
    channel_dataframe,
    describe_probe,
)
from guppy.utils.dandi_preview import (
    AssetPreview,
    ChannelInfo,
    ExampleTraces,
    PhotometryProbe,
    SeriesInfo,
)


def make_probe(*, channel_count=2, has_photometry=True):
    """Build a probe of one two-channel series recorded in the VTA, or of an empty file."""
    if not has_photometry:
        return PhotometryProbe(session_description="behavior only", event_names=("nose_poke",))
    channels = tuple(
        ChannelInfo(
            store_name=f"fiber_photometry_response_series_{index}",
            series_name="fiber_photometry_response_series",
            column_index=index,
            location="VTA",
            indicator="GCaMP7b",
            excitation_wavelength_in_nm=405.0 if index == 0 else 465.0,
            emission_wavelength_in_nm=525.0,
            suggested_label="control_VTA" if index == 0 else "signal_VTA",
        )
        for index in range(channel_count)
    )
    return PhotometryProbe(
        series=(
            SeriesInfo(
                name="fiber_photometry_response_series",
                sample_count=3000,
                channel_count=channel_count,
                sampling_rate_in_hz=30.0,
                duration_in_seconds=100.0,
            ),
        ),
        channels=channels,
        event_names=("nose_poke",),
        session_description="A reward task",
        session_start_time="2024-01-01T00:00:00+00:00",
        identifier="session-1",
        subject={"subject_id": "mouse-1", "species": "Mus musculus", "sex": "F"},
    )


def make_preview(*, probe=None, series_name="fiber_photometry_response_series", with_traces=True):
    """Build an AssetPreview, optionally carrying example traces for ``series_name``."""
    probe = probe if probe is not None else make_probe()
    traces = None
    if with_traces and probe.has_photometry:
        timestamps = np.linspace(0.0, 10.0, 50)
        traces = ExampleTraces(
            series_name=series_name,
            timestamps=timestamps,
            traces={channel.store_name: np.sin(timestamps) for channel in probe.channels},
        )
    return AssetPreview(dandiset_id="000001", asset_path="sub-01/ses-1.nwb", probe=probe, traces=traces)


class RecordingPreview:
    """Stand-in for ``preview_asset`` that records its calls and returns a canned preview."""

    def __init__(self, preview=None):
        self.preview = preview if preview is not None else make_preview()
        self.calls = []

    def __call__(self, *, dandiset_id, asset_path, series_name=None):
        self.calls.append(
            {
                "dandiset_id": dandiset_id,
                "asset_path": asset_path,
                "series_name": series_name,
            }
        )
        return self.preview


class TestChannelDataframe:
    def test_one_row_per_channel(self):
        frame = channel_dataframe(make_probe())
        assert list(frame.columns) == list(CHANNEL_COLUMNS)
        assert list(frame["Store name"]) == [
            "fiber_photometry_response_series_0",
            "fiber_photometry_response_series_1",
        ]
        assert list(frame["Suggested label"]) == ["control_VTA", "signal_VTA"]

    def test_missing_fields_render_as_blanks(self):
        probe = PhotometryProbe(
            channels=(
                ChannelInfo(
                    store_name="series",
                    series_name="series",
                    column_index=None,
                    location=None,
                    indicator=None,
                    excitation_wavelength_in_nm=None,
                    emission_wavelength_in_nm=None,
                    suggested_label=None,
                ),
            )
        )
        frame = channel_dataframe(probe)
        assert list(frame.loc[0]) == ["series", "", "", "", "", ""]


class TestDescribeProbe:
    def test_reports_every_series_with_its_timing(self):
        text = describe_probe(preview=make_preview())
        assert "`sub-01/ses-1.nwb`" in text
        assert "2 channel(s)" in text
        assert "3,000 samples" in text
        assert "30.00 Hz" in text
        assert "1.7 min" in text

    def test_reports_subject_and_session(self):
        text = describe_probe(preview=make_preview())
        assert "mouse-1 · Mus musculus · F" in text
        assert "**Session:** A reward task" in text
        assert "**Event objects:** nose_poke" in text

    def test_file_without_photometry_is_called_out(self):
        text = describe_probe(preview=make_preview(probe=make_probe(has_photometry=False)))
        assert "no `FiberPhotometryResponseSeries`" in text
        assert "the recordings are the large ones" in text
        assert "**Event objects:** nose_poke" in text

    def test_file_with_neither_photometry_nor_events_says_only_that(self):
        preview = make_preview(probe=PhotometryProbe(session_description="empty"))
        text = describe_probe(preview=preview)
        assert "no `FiberPhotometryResponseSeries`" in text
        assert "Event objects" not in text

    def test_irregular_timing_is_named_rather_than_faked(self):
        probe = PhotometryProbe(
            series=(
                SeriesInfo(
                    name="series",
                    sample_count=10,
                    channel_count=1,
                    sampling_rate_in_hz=None,
                    duration_in_seconds=None,
                ),
            )
        )
        text = describe_probe(preview=make_preview(probe=probe, with_traces=False))
        assert "irregular timestamps" in text
        assert "unknown length" in text


class TestBuildTraceOverlay:
    def test_one_curve_per_channel_over_a_shared_time_axis(self):
        overlay = build_trace_overlay(preview=make_preview())
        assert isinstance(overlay, hv.NdOverlay)
        assert list(overlay.keys()) == [
            "fiber_photometry_response_series_0",
            "fiber_photometry_response_series_1",
        ]
        curve = overlay["fiber_photometry_response_series_0"]
        np.testing.assert_allclose(curve.dimension_values(0), np.linspace(0.0, 10.0, 50))

    def test_title_names_the_series_and_the_window(self):
        overlay = build_trace_overlay(preview=make_preview())
        assert overlay.opts.get("plot").kwargs["title"] == "First 10 s of fiber_photometry_response_series"


@pytest.fixture
def preview_panel(panel_extension):
    return DandiPreviewPanel(preview_function=RecordingPreview())


class TestDandiPreviewPanel:
    def test_starts_hidden_and_empty(self, preview_panel):
        assert preview_panel.panel.visible is False
        assert preview_panel.preview is None
        assert preview_panel.channel_table.value.empty

    def test_show_fills_every_part(self, preview_panel):
        preview_panel.show(preview=make_preview())
        assert preview_panel.panel.visible is True
        assert "2 channel(s)" in preview_panel.summary.object
        assert len(preview_panel.channel_table.value) == 2
        assert isinstance(preview_panel.trace_pane.object, hv.NdOverlay)

    def test_single_series_file_hides_the_series_picker(self, preview_panel):
        preview_panel.show(preview=make_preview())
        assert preview_panel.series_select.visible is False
        assert preview_panel.series_select.value == "fiber_photometry_response_series"

    def test_multi_series_file_shows_the_series_picker(self, preview_panel):
        probe = make_probe()
        extra = SeriesInfo(
            name="second_series",
            sample_count=10,
            channel_count=1,
            sampling_rate_in_hz=30.0,
            duration_in_seconds=1.0,
        )
        probe = PhotometryProbe(
            series=(*probe.series, extra),
            channels=probe.channels,
            event_names=probe.event_names,
        )
        preview_panel.show(preview=make_preview(probe=probe))
        assert preview_panel.series_select.visible is True
        assert preview_panel.series_select.options == [
            "fiber_photometry_response_series",
            "second_series",
        ]

    def test_showing_a_preview_does_not_restream_it(self, preview_panel):
        preview_panel.show(preview=make_preview())
        assert preview_panel.preview_function.calls == []

    def test_picking_another_series_restreams_that_series(self, preview_panel):
        second = SeriesInfo(
            name="second_series",
            sample_count=10,
            channel_count=1,
            sampling_rate_in_hz=30.0,
            duration_in_seconds=1.0,
        )
        probe = PhotometryProbe(series=(*make_probe().series, second), channels=make_probe().channels)
        preview_panel.preview_function.preview = make_preview(probe=probe, series_name="second_series")
        preview_panel.show(preview=make_preview(probe=probe))

        preview_panel.series_select.value = "second_series"

        assert preview_panel.preview_function.calls == [
            {
                "dandiset_id": "000001",
                "asset_path": "sub-01/ses-1.nwb",
                "series_name": "second_series",
            }
        ]
        assert preview_panel.preview.traces.series_name == "second_series"

    def test_file_without_photometry_hides_the_channel_table_and_plot(self, preview_panel):
        preview_panel.show(preview=make_preview(probe=make_probe(has_photometry=False)))
        assert preview_panel.panel.visible is True
        assert preview_panel.channel_table.visible is False
        assert preview_panel.trace_pane.object is None

    def test_clear_empties_and_hides(self, preview_panel):
        preview_panel.show(preview=make_preview())
        preview_panel.clear()
        assert preview_panel.panel.visible is False
        assert preview_panel.preview is None
        assert preview_panel.summary.object == ""
        assert preview_panel.channel_table.value.empty
        assert preview_panel.trace_pane.object is None
