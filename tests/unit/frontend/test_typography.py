from panel.viewable import Viewable

from guppy.frontend.typography import (
    BODY_SIZE,
    COMPONENT_CSS,
    HEADING_SIZE,
    REGULAR,
    SEMIBOLD,
    SMALL_SIZE,
    TITLE_SIZE,
    GuppyTypeDesign,
)


class TestScale:
    def test_four_sizes(self):
        assert (TITLE_SIZE, HEADING_SIZE, BODY_SIZE, SMALL_SIZE) == ("20px", "16px", "14px", "12px")

    def test_two_weights(self):
        assert (REGULAR, SEMIBOLD) == (400, 600)


class TestComponentCss:
    def test_body_text_is_14px(self):
        assert ":host {\n  --bokeh-base-font:" in COMPONENT_CSS
        assert "  font-size: 14px;\n  line-height: 1.45;" in COMPONENT_CSS

    def test_card_titles_are_16px_semibold(self):
        assert ".card-header, :host(.card-title) h3 {\n  font-size: 16px;\n  font-weight: 600;\n}" in COMPONENT_CSS

    def test_section_titles_are_14px_semibold(self):
        assert "h3 {\n  font-size: 14px;\n  font-weight: 600;" in COMPONENT_CSS

    def test_small_text_is_12px_in_the_secondary_gray(self):
        assert ":host(.small), :host(.small) p {\n  font-size: 12px;\n  color: #6C757D;\n}" in COMPONENT_CSS


class TestGuppyTypeDesign:
    def test_adds_the_type_stylesheet_to_every_component(self):
        assert GuppyTypeDesign.modifiers[Viewable]["stylesheets"][1] == COMPONENT_CSS
