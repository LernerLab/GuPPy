import pytest

from guppy.frontend.tab_opener import TabOpener


@pytest.fixture
def tab_opener(panel_extension):
    return TabOpener()


@pytest.fixture
def sent(tab_opener, monkeypatch):
    """Record the messages bound for the browser, which does not exist outside a served session."""
    messages = []
    monkeypatch.setattr(tab_opener, "_send_msg", messages.append)
    return messages


class TestTabOpener:
    def test_navigate_points_the_placeholder_at_the_page(self, tab_opener, sent):
        tab_opener.navigate(key="open_label_stores", href="http://localhost:5100/")

        assert sent == [{"key": "open_label_stores", "href": "http://localhost:5100/"}]

    def test_close_closes_the_placeholder(self, tab_opener, sent):
        tab_opener.close(key="open_label_stores")

        assert sent == [{"key": "open_label_stores", "href": None}]
