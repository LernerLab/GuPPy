"""The homepage's main area, which shows the input parameters and every page a step opens."""

import panel as pn

HOME_TAB_NAME = "Input Parameters"

# Panel makes either every tab closable or none, so the form's tab, always first, hides its button.
_KEEP_HOME_TAB_OPEN = ".bk-tab:first-child .bk-close { display: none; }"


class PageArea:
    """Tabs holding the input parameters form and one closable tab per page a sidebar button opened.

    Parameters
    ----------
    home : pn.viewable.Viewable
        The input parameters form, shown in the first tab, which cannot be closed.
    """

    def __init__(self, *, home: pn.viewable.Viewable) -> None:
        self.home = _scrolling(home)
        # Filling the main area below the header, with each tab scrolling on its own, keeps the tab
        # bar in view and every tab where the user last left it.
        self.tabs = pn.Tabs(
            (HOME_TAB_NAME, self.home),
            closable=True,
            sizing_mode="stretch_width",
            styles={"height": "calc(100vh - 100px)"},
            stylesheets=[_KEEP_HOME_TAB_OPEN],
        )
        # The page each key's tab shows, so opening a key again replaces its tab.
        self._pages: dict[str, pn.viewable.Viewable] = {}

    def show(self, *, key: str, name: str, content: pn.viewable.Viewable) -> None:
        """Show a page in the tab for ``key`` and switch to it.

        Parameters
        ----------
        key : str
            Identifies what opened the page, such as its sidebar button; the page replaces the
            one ``key`` opened before, if that tab is still open.
        name : str
            The tab's label.
        content : pn.viewable.Viewable
            The page.
        """
        content = _scrolling(content)
        previous = self._pages.get(key)
        indices = [index for index, page in enumerate(self.tabs.objects) if page is previous]
        self._pages[key] = content
        if indices:
            (index,) = indices
            self.tabs[index] = (name, content)
        else:
            self.tabs.append((name, content))
            index = len(self.tabs) - 1
        self.tabs.active = index


def _scrolling(content: pn.viewable.Viewable) -> pn.Column:
    return pn.Column(content, scroll=True, sizing_mode="stretch_both")
