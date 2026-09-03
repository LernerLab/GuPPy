import holoviews as hv
import numpy as np
import pytest
from bokeh.models import BoxSelectTool, Plot

from guppy.visualization.preprocessing import (
    build_control_signal_fit,
    build_markable_trace,
    make_spans_pipe,
    set_drag_gesture,
)
from guppy_test_data import resolve_plot


def image_extent(image):
    """Return an image's ``(left, bottom, right, top)`` bounds as plain floats.

    A shaded trace is an ``hv.RGB``, so its data is pixels rather than samples; the bounds
    are what carry the extent of the trace that was drawn into it.
    """
    return tuple(float(value) for value in image.bounds.lbrt())


def bounds_streams_created_by(build):
    """Run ``build`` and return the plot with the ``BoundsX`` streams it registered.

    Streams register themselves against their source in a process-wide registry, so the
    ones this call added are found by differencing the registry around it. The snapshot
    holds the stream objects themselves rather than their ids, because a stream left by
    an earlier test can be collected and its id handed to one of ours.
    """

    def registered():
        return [
            stream
            for streams in hv.streams.Stream.registry.values()
            for stream in streams
            if isinstance(stream, hv.streams.BoundsX)
        ]

    before = registered()
    plot = build()
    added = [stream for stream in registered() if not any(stream is seen for seen in before)]
    return plot, added


@pytest.fixture
def timestamps():
    return np.array([0.0, 1.0, 2.0, 3.0, 4.0])


class TestBuildControlSignalFit:
    # Each trace spans a distinct range so the bounds of a shaded panel identify which
    # trace was drawn into it.
    @pytest.fixture
    def control(self):
        return np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    @pytest.fixture
    def signal_trace(self):
        return np.array([10.0, 11.0, 12.0, 13.0, 14.0])

    @pytest.fixture
    def fit(self):
        return np.array([20.0, 21.0, 22.0, 23.0, 24.0])

    def test_returns_three_stacked_elements(self, panel_extension, timestamps, control, signal_trace, fit):
        layout = build_control_signal_fit(
            x=timestamps,
            control=control,
            signal=signal_trace,
            fit=fit,
            titles=["control", "signal", "fit"],
            suptitle="session",
            spans=make_spans_pipe(windows=[]),
        )
        assert isinstance(layout, hv.Layout)
        elements = layout.values()
        assert len(elements) == 3

    def test_bottom_axis_overlays_signal_and_fit(self, panel_extension, timestamps, control, signal_trace, fit):
        layout = build_control_signal_fit(
            x=timestamps,
            control=control,
            signal=signal_trace,
            fit=fit,
            titles=["control", "signal", "fit"],
            suptitle="session",
            spans=make_spans_pipe(windows=[]),
        )
        # Top two carry a single shaded trace plus the (empty) spans layer; the bottom
        # also overlays the fit.
        control_overlay, signal_overlay, fit_overlay = (resolve_plot(plot) for plot in layout.values())
        control_image, _ = control_overlay.values()
        signal_image, _ = signal_overlay.values()
        assert image_extent(control_image) == (0.0, 1.0, 4.0, 5.0)
        assert image_extent(signal_image) == (0.0, 10.0, 4.0, 14.0)

        shaded_signal, shaded_fit, _ = fit_overlay.values()
        assert image_extent(shaded_signal) == (0.0, 10.0, 4.0, 14.0)
        assert image_extent(shaded_fit) == (0.0, 20.0, 4.0, 24.0)

    def test_extra_traces_stack_below_the_fit(self, panel_extension, timestamps, control, signal_trace, fit):
        layout = build_control_signal_fit(
            x=timestamps,
            control=control,
            signal=signal_trace,
            fit=fit,
            titles=["control", "signal", "fit"],
            suptitle="session",
            spans=make_spans_pipe(windows=[]),
            extra_traces={
                "z_score": np.array([30.0, 31.0, 32.0, 33.0, 34.0]),
                "dff": np.array([40.0, 41.0, 42.0, 43.0, 44.0]),
            },
        )
        overlays = [resolve_plot(plot) for plot in layout.values()]
        assert len(overlays) == 5
        z_score_image, _ = overlays[3].values()
        dff_image, _ = overlays[4].values()
        assert image_extent(z_score_image) == (0.0, 30.0, 4.0, 34.0)
        assert image_extent(dff_image) == (0.0, 40.0, 4.0, 44.0)

    def test_extra_traces_are_shaded_by_the_same_windows(self, panel_extension, timestamps, control, signal_trace, fit):
        layout = build_control_signal_fit(
            x=timestamps,
            control=control,
            signal=signal_trace,
            fit=fit,
            titles=["control", "signal", "fit"],
            suptitle="session",
            spans=make_spans_pipe(windows=[(1.0, 2.0)]),
            extra_traces={"z_score": np.array([30.0, 31.0, 32.0, 33.0, 34.0])},
        )
        spans = [item for item in resolve_plot(layout.values()[3]).values() if isinstance(item, hv.VSpans)][0]
        np.testing.assert_array_equal(spans.dimension_values("x0"), np.array([1.0]))
        np.testing.assert_array_equal(spans.dimension_values("x1"), np.array([2.0]))

    def test_omitting_spans_draws_the_traces_unshaded(self, panel_extension, timestamps, control, signal_trace, fit):
        layout = build_control_signal_fit(
            x=timestamps,
            control=control,
            signal=signal_trace,
            fit=fit,
            titles=["control", "signal", "fit"],
            suptitle="session",
        )
        # Without the spans layer a single-trace panel is a bare image; only the fit panel,
        # which overlays two traces, stays an Overlay.
        control_image, signal_image, fit_overlay = (resolve_plot(plot) for plot in layout.values())
        assert isinstance(control_image, hv.RGB)
        assert isinstance(signal_image, hv.RGB)
        assert [type(item).__name__ for item in fit_overlay.values()] == ["RGB", "RGB"]

    def test_windows_shade_all_three_axes(self, panel_extension, timestamps, control, signal_trace, fit):
        layout = build_control_signal_fit(
            x=timestamps,
            control=control,
            signal=signal_trace,
            fit=fit,
            titles=["control", "signal", "fit"],
            suptitle="session",
            spans=make_spans_pipe(windows=[(1.0, 2.0), (3.0, 4.0)]),
        )
        for plot in layout.values():
            element = resolve_plot(plot)
            assert isinstance(element, hv.Overlay)
            spans = [item for item in element.values() if isinstance(item, hv.VSpans)]
            assert len(spans) == 1
            np.testing.assert_array_equal(spans[0].dimension_values("x0"), np.array([1.0, 3.0]))
            np.testing.assert_array_equal(spans[0].dimension_values("x1"), np.array([2.0, 4.0]))

    def test_sending_on_the_pipe_repaints_spans_without_rebuilding_the_layout(
        self, panel_extension, timestamps, control, signal_trace, fit
    ):
        spans = make_spans_pipe(windows=[])
        layout = build_control_signal_fit(
            x=timestamps,
            control=control,
            signal=signal_trace,
            fit=fit,
            titles=["control", "signal", "fit"],
            suptitle="session",
            spans=spans,
        )

        spans.send([(1.0, 2.0)])

        for plot in layout.values():
            span_element = [item for item in resolve_plot(plot).values() if isinstance(item, hv.VSpans)][0]
            np.testing.assert_array_equal(span_element.dimension_values("x0"), np.array([1.0]))


class TestPanelSizing:
    """Every panel must render at the same size, spans layer or not.

    The fit panel is an Overlay rather than a single curve, and options set on an Overlay
    are dropped when the spans layer composes it into a new one — which silently left it
    at bokeh's 300x300 default next to its fixed-size neighbours.
    """

    @pytest.fixture
    def rendered_plots(self, panel_extension):
        from bokeh.models import Plot

        def render(windows):
            x = np.arange(0.0, 50.0, 0.1)
            layout = build_control_signal_fit(
                x=x,
                control=np.sin(x),
                signal=np.cos(x),
                fit=np.cos(x) * 0.9,
                titles=["control", "signal", "fit"],
                suptitle="session",
                spans=make_spans_pipe(windows=windows),
                extra_traces={"z_score": np.sin(x) * 2},
            )
            figure = hv.render(layout)
            return [model for model in figure.references() if isinstance(model, Plot)]

        return render

    @pytest.mark.parametrize("windows", [[], [(10.0, 20.0)]], ids=["no_spans", "with_spans"])
    def test_every_panel_stretches_to_the_page_width(self, rendered_plots, windows):
        plots = rendered_plots(windows)
        assert len(plots) == 4
        assert {(plot.sizing_mode, plot.width, plot.height) for plot in plots} == {("stretch_width", None, 220)}

    def test_every_panel_keeps_its_title(self, rendered_plots):
        titles = {plot.title.text for plot in rendered_plots([])}
        assert titles == {"session — control", "signal", "fit", "z_score"}


class TestBuildMarkableTrace:
    """The marking page draws one trace at a time, and a drag across it reports its bounds."""

    @pytest.fixture
    def build(self, panel_extension):
        def build(on_x_select=lambda start, end: None, overlay=None, hooks=None):
            x = np.arange(0.0, 50.0, 0.1)
            return build_markable_trace(
                x=x,
                values=np.sin(x),
                overlay=overlay,
                title="session — signal",
                spans=make_spans_pipe(windows=[(1.0, 2.0)]),
                on_x_select=on_x_select,
                hooks=hooks,
            )

        return build

    @pytest.fixture
    def rendered(self, build):
        def rendered(**kwargs):
            plots = [model for model in hv.render(build(**kwargs)).references() if isinstance(model, Plot)]
            assert len(plots) == 1
            return plots[0]

        return rendered

    def test_draws_a_single_panel(self, rendered):
        plot = rendered()

        assert (plot.title.text, plot.sizing_mode, plot.width, plot.height) == (
            "session — signal",
            "stretch_width",
            None,
            420,
        )

    def test_shades_the_windows_it_was_given(self, build):
        span_element = [item for item in resolve_plot(build()).values() if isinstance(item, hv.VSpans)][0]

        np.testing.assert_array_equal(span_element.dimension_values("x0"), np.array([1.0]))
        np.testing.assert_array_equal(span_element.dimension_values("x1"), np.array([2.0]))

    def test_a_drag_reports_the_bounds_it_covered(self, build):
        dragged = []
        _, streams = bounds_streams_created_by(lambda: build(lambda start, end: dragged.append((start, end))))

        assert len(streams) == 1
        streams[0].event(boundsx=(3.0, 7.0))

        assert dragged == [(3.0, 7.0)]

    def test_offers_a_horizontal_select_tool(self, rendered):
        select_tools = [tool for tool in rendered().toolbar.tools if isinstance(tool, BoxSelectTool)]

        assert [tool.dimensions for tool in select_tools] == ["width"]

    def test_dragging_marks_and_scrolling_still_zooms(self, rendered):
        """Naming the select tool must not cost the wheel zoom that comes with it."""
        toolbar = rendered().toolbar

        assert type(toolbar.active_drag).__name__ == "BoxSelectTool"
        assert type(toolbar.active_scroll).__name__ == "WheelZoomTool"

    def test_overlay_is_drawn_over_the_trace(self, build):
        x = np.arange(0.0, 50.0, 0.1)
        images = [item for item in resolve_plot(build(overlay=np.cos(x))).values() if isinstance(item, hv.RGB)]

        assert len(images) == 2

    def test_hooks_receive_the_rendered_figure(self, build):
        captured = []
        plot = build(hooks=[lambda plot, element: captured.append(plot.state)])
        figure = hv.render(plot)

        assert captured and all(state is figure for state in captured)


class TestSetDragGesture:
    """Switching modes re-arms the live figure instead of redrawing it, so zoom survives."""

    @pytest.fixture
    def figure(self, panel_extension):
        x = np.arange(0.0, 50.0, 0.1)
        plot = build_markable_trace(
            x=x,
            values=np.sin(x),
            title="session — signal",
            spans=make_spans_pipe(windows=[]),
            on_x_select=lambda start, end: None,
        )
        return [model for model in hv.render(plot).references() if isinstance(model, Plot)][0]

    def test_navigating_arms_panning(self, figure):
        set_drag_gesture(figure=figure, marking=False)

        assert type(figure.toolbar.active_drag).__name__ == "PanTool"

    def test_marking_arms_the_select_tool(self, figure):
        set_drag_gesture(figure=figure, marking=False)

        set_drag_gesture(figure=figure, marking=True)

        assert type(figure.toolbar.active_drag).__name__ == "BoxSelectTool"

    def test_the_zoomed_range_is_left_alone(self, figure):
        """The view the user navigated to is what a mode switch must not throw away."""
        figure.x_range.start, figure.x_range.end = 10.0, 20.0

        set_drag_gesture(figure=figure, marking=True)

        assert (figure.x_range.start, figure.x_range.end) == (10.0, 20.0)

    def test_scrolling_keeps_zooming_in_either_mode(self, figure):
        for marking in (False, True):
            set_drag_gesture(figure=figure, marking=marking)

            assert type(figure.toolbar.active_scroll).__name__ == "WheelZoomTool"
