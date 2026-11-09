"""A plot beside a side panel, split by a handle that drags to resize the panel and a button that hides it."""

from typing import ClassVar

import panel as pn
import param
from panel.custom import JSComponent

from .typography import INK, SECONDARY_INK

_RULE_COLOR = "#E9ECEF"
_HANDLE_HOVER_COLOR = "#ADB5BD"
_BORDER_COLOR = "#C8CCD0"

# The narrowest the side panel goes, and the least width a drag leaves the plot.
MINIMUM_SIDE_WIDTH = 220
MINIMUM_PLOT_WIDTH = 300


class SidePanelHandle(JSComponent):
    """The thin divider between a plot and its side panel.

    Dragging it sets ``side_width``; its button flips ``collapsed``. It holds no content, so
    the plot and the panel stay ordinary Panel layouts that size themselves.
    """

    side_width = param.Integer(default=460, bounds=(MINIMUM_SIDE_WIDTH, None))
    collapsed = param.Boolean(default=False)
    side_label = param.String(default="")

    _esm = """
    export function render({ model }) {
      const handle = document.createElement("div");
      handle.className = "handle";
      const toggle = document.createElement("button");
      toggle.className = "toggle";
      handle.append(toggle);

      function update() {
        handle.classList.toggle("collapsed", model.collapsed);
        toggle.textContent = model.collapsed ? "\\u2039" : "\\u203A";
        toggle.title = `${model.collapsed ? "Show" : "Hide"} ${model.side_label}`;
      }
      update();
      model.on("collapsed", update);

      toggle.addEventListener("pointerdown", (event) => event.stopPropagation());
      toggle.addEventListener("click", () => { model.collapsed = !model.collapsed; });

      handle.addEventListener("pointerdown", (event) => {
        if (model.collapsed) {
          return;
        }
        event.preventDefault();
        const startX = event.clientX;
        const startWidth = model.side_width;
        // The handle's row spans the plot, the handle and the panel; a drag stops short of
        // squeezing the plot below its minimum width.
        const row = handle.getRootNode().host.getRootNode().host;
        const widest = Math.max(MINIMUM_SIDE_WIDTH, row.getBoundingClientRect().width - MINIMUM_PLOT_WIDTH);
        let pending = null;
        handle.setPointerCapture(event.pointerId);
        handle.classList.add("dragging");
        const move = (moveEvent) => {
          pending = Math.round(Math.max(MINIMUM_SIDE_WIDTH, Math.min(widest, startWidth + startX - moveEvent.clientX)));
          requestAnimationFrame(() => {
            if (pending !== null) {
              model.side_width = pending;
              pending = null;
            }
          });
        };
        const release = (releaseEvent) => {
          handle.releasePointerCapture(releaseEvent.pointerId);
          handle.classList.remove("dragging");
          handle.removeEventListener("pointermove", move);
          handle.removeEventListener("pointerup", release);
        };
        handle.addEventListener("pointermove", move);
        handle.addEventListener("pointerup", release);
      });
      return handle;
    }
    """.replace(
        "MINIMUM_SIDE_WIDTH", str(MINIMUM_SIDE_WIDTH)
    ).replace(
        "MINIMUM_PLOT_WIDTH", str(MINIMUM_PLOT_WIDTH)
    )

    _stylesheets: ClassVar[list[str]] = [
        f"""
        .handle {{
          position: relative; height: 100%; min-height: 60px; display: flex; justify-content: center;
          cursor: col-resize; touch-action: none;
        }}
        .handle::before {{ content: ""; width: 1px; height: 100%; background: {_RULE_COLOR}; }}
        .handle:hover::before, .handle.dragging::before {{ width: 3px; background: {_HANDLE_HOVER_COLOR}; }}
        .handle.collapsed {{ cursor: default; }}
        .toggle {{
          position: absolute; top: 4px; width: 20px; height: 28px; padding: 0; cursor: pointer;
          border: 1px solid {_BORDER_COLOR}; border-radius: 4px; background: White;
          color: {SECONDARY_INK}; font-size: 16px; line-height: 1;
        }}
        .toggle:hover {{ border-color: {_HANDLE_HOVER_COLOR}; color: {INK}; }}
        """
    ]


class SidePanelSplit(pn.viewable.Viewer):
    """The plot on the left and a side panel on the right, with a draggable handle between them.

    Dragging the handle resizes the side panel, which the plot gives up width to or takes it
    back from. The button on the handle hides the side panel so the plot fills the width, and
    shows it again.

    Parameters
    ----------
    main : pn.viewable.Viewable
        The plot, which takes the width the side panel leaves.
    side : pn.viewable.Viewable
        The side panel's content, stretched to its width.
    side_label : str
        What the side panel holds (e.g. "Events"), named in the button's tooltip.
    height : int or bound function
        The plot's height in pixels, which the handle matches.
    side_width : int, optional
        The side panel's starting width in pixels.
    """

    def __init__(
        self,
        *,
        main: pn.viewable.Viewable,
        side: pn.viewable.Viewable,
        side_label: str,
        height: object,
        side_width: int = 460,
    ) -> None:
        super().__init__()
        self.handle = SidePanelHandle(
            side_width=side_width, side_label=side_label, width=14, height=height, margin=(8, 0, 10, 0)
        )
        self.side = pn.Column(side, width=side_width, margin=0)
        self.handle.param.watch(self._resize, "side_width")
        self.handle.param.watch(self._show_or_hide, "collapsed")
        self._layout = pn.Row(main, self.handle, self.side, sizing_mode="stretch_width")

    def _resize(self, event: param.parameterized.Event) -> None:
        self.side.width = event.new

    def _show_or_hide(self, event: param.parameterized.Event) -> None:
        self.side.visible = not event.new

    def __panel__(self) -> pn.Row:
        return self._layout
