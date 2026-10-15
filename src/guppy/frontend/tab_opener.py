"""Opening a page in the browser tab the user's click already opened."""

import panel as pn
from panel.custom import JSComponent

# Browsers only let a page open a new tab during a user's click, and the page a click asks for is
# built on the server afterwards. So the click itself opens a placeholder tab in the browser, and
# the server later points it at the page with TabOpener, or closes it when nothing came of the
# click. The placeholders are kept on window, under the key of the button that opened them.
PLACEHOLDER_TAB_CODE = """
const tab = window.open("", "_blank");
if (tab !== null) {
  tab.document.title = "Opening…";
  tab.document.body.innerHTML = '<p style="font-family: sans-serif; padding: 1em">Opening the GuPPy page…</p>';
}
window.guppyPendingTabs = window.guppyPendingTabs || {};
window.guppyPendingTabs[key] = tab;
"""


def open_placeholder_tab_on_click(*, button: pn.widgets.Button, key: str) -> None:
    """Make every click on ``button`` open a placeholder tab for :class:`TabOpener` to fill.

    Parameters
    ----------
    button : pn.widgets.Button
        The button whose click asks for a page.
    key : str
        Names the placeholder, for :meth:`TabOpener.navigate` and :meth:`TabOpener.close`.
    """
    button.js_on_click(code=f"const key = {key!r};\n{PLACEHOLDER_TAB_CODE}")


class TabOpener(JSComponent):
    """Zero-size homepage component that fills or closes the placeholder tab a click opened."""

    _esm = """
    export function render({ model }) {
      model.on("msg:custom", (message) => {
        const pendingTabs = window.guppyPendingTabs || {}
        const tab = pendingTabs[message.key]
        delete pendingTabs[message.key]
        if (tab == null || tab.closed) {
          return
        }
        if (message.href === null) {
          tab.close()
        } else {
          tab.location.href = new URL(message.href, window.location.href).href
        }
      })
    }
    """

    def __init__(self, **params: object) -> None:
        super().__init__(width=0, height=0, margin=0, **params)

    def navigate(self, *, key: str, href: str) -> None:
        """Show the page at ``href`` in the placeholder tab opened under ``key``.

        Parameters
        ----------
        key : str
            The placeholder's key.
        href : str
            Where the page is, either absolute or relative to the homepage.
        """
        self._send_msg({"key": key, "href": href})

    def close(self, *, key: str) -> None:
        """Close the placeholder tab opened under ``key``.

        Parameters
        ----------
        key : str
            The placeholder's key.
        """
        self._send_msg({"key": key, "href": None})
