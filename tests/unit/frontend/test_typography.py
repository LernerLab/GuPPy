import base64

from panel.viewable import Viewable

from guppy.frontend.typography import (
    BODY_SIZE,
    COMPONENT_CSS,
    FONTS_DIRECTORY,
    HEADING_SIZE,
    REGULAR,
    SEMIBOLD,
    SMALL_SIZE,
    TEXT_FAMILY,
    TITLE_SIZE,
    GuppyTypeDesign,
    font_face_css,
    title_case,
)
from guppy.visualization.palette import PLOT_THEME


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

    def test_plots_in_the_page_take_the_plot_text(self):
        assert GuppyTypeDesign._themes["default"].bokeh_theme is PLOT_THEME


class TestInter:
    def test_text_is_set_in_inter_first(self):
        assert TEXT_FAMILY.startswith('"Inter", ')

    def test_regular_and_semibold_ship_with_their_license(self):
        assert sorted(path.name for path in FONTS_DIRECTORY.iterdir()) == [
            "Inter-LICENSE.txt",
            "Inter-Regular.woff2",
            "Inter-SemiBold.woff2",
        ]

    def test_font_faces_embed_both_weights(self):
        css = font_face_css()
        regular = base64.b64encode((FONTS_DIRECTORY / "Inter-Regular.woff2").read_bytes()).decode("ascii")
        semibold = base64.b64encode((FONTS_DIRECTORY / "Inter-SemiBold.woff2").read_bytes()).decode("ascii")

        assert css.count("@font-face") == 2
        assert f'font-weight: 400; font-display: block; src: url("data:font/woff2;base64,{regular}")' in css
        assert f'font-weight: 600; font-display: block; src: url("data:font/woff2;base64,{semibold}")' in css

    def test_tabular_figures_apply_to_table_cells_only(self):
        assert ".tabulator-cell, table.panel-df td {\n  font-variant-numeric: tabular-nums;\n}" in COMPONENT_CSS
        assert COMPONENT_CSS.count("tabular-nums") == 1


class TestTitleCase:
    def test_capitalizes_each_word(self):
        assert title_case("Optical fiber models") == "Optical Fiber Models"

    def test_keeps_minor_words_lowercase_inside_the_name(self):
        assert title_case("Member runs to average") == "Member Runs to Average"

    def test_capitalizes_a_minor_word_at_either_end(self):
        assert title_case("Control for") == "Control For"

    def test_capitalizes_each_part_of_a_hyphenated_word(self):
        assert title_case("Fiber-photometry channels") == "Fiber-Photometry Channels"

    def test_keeps_words_that_carry_capitals(self):
        assert title_case("Core NWB metadata") == "Core NWB Metadata"

    def test_leaves_parentheses_as_written(self):
        assert title_case("Display settings (all plots)") == "Display Settings (all plots)"
