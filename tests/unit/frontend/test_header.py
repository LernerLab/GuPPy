from pathlib import Path

import pytest

import guppy
from guppy.frontend.header import HEADER_CSS, LOGO_GRADIENT, guppy_header_template
from guppy.frontend.typography import GuppyTypeDesign, font_face_css


@pytest.fixture
def template(panel_extension):
    return guppy_header_template(page_name="Label Stores — mouse_01", title="Label Stores GUI - mouse_01")


class TestGuppyHeaderTemplate:
    def test_title_names_the_browser_tab(self, template):
        assert template.title == "Label Stores GUI - mouse_01"

    def test_logo_and_favicon_are_the_packaged_guppy_assets(self, template):
        assets = Path(guppy.__file__).parent / "assets"

        assert (Path(template.logo), Path(template.favicon)) == (assets / "GuppyMark.png", assets / "favicon.png")

    def test_header_is_white_with_dark_text(self, template):
        assert (template.header_background, template.header_color) == ("#FFFFFF", "#212529")

    def test_header_shows_the_wordmark_and_page_name(self, template):
        (wordmark,) = template.header.objects

        assert 'alt="GuPPy"' in wordmark.object
        assert ">Label Stores — mouse_01</span>" in wordmark.object

    def test_page_loads_inter_and_applies_the_type_design(self, template):
        assert font_face_css() in template.config.raw_css
        assert template.design is GuppyTypeDesign

    def test_header_has_the_rainbow_rule(self, template):
        assert HEADER_CSS in template.config.raw_css
        assert f"border-image: {LOGO_GRADIENT} 1;" in HEADER_CSS
