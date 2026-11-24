from guppy.frontend.tile_toggle import Tile, tile_toggle_stylesheet

FIRST = Tile(icon="M1 1h2", accent="#111111", description="The first option")
SECOND = Tile(icon="M2 2h4", accent="#222222", description="The second option")


class TestTileToggleStylesheet:
    def test_each_option_gets_its_own_icon_accent_and_description(self):
        stylesheet = tile_toggle_stylesheet(tiles=[FIRST, SECOND])

        assert (
            ":host .bk-btn-group > button:nth-child(1) {\n"
            "  --icon: url(\"data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
            "fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E"
            "%3Cpath d='M1 1h2'/%3E%3C/svg%3E\");\n"
            "  --accent: #111111;\n"
            '  --description: "The first option";\n'
            "}"
        ) in stylesheet
        assert "  --accent: #222222;\n" '  --description: "The second option";\n' in stylesheet

    def test_full_tiles_show_the_description_under_the_name(self):
        stylesheet = tile_toggle_stylesheet(tiles=[FIRST, SECOND])

        assert "content: var(--description);" in stylesheet
        assert "height: 35px" not in stylesheet

    def test_compact_tiles_match_a_dropdown_and_show_no_description(self):
        stylesheet = tile_toggle_stylesheet(tiles=[Tile(icon="M1 1h2", accent="#111111")], compact=True)

        assert "height: 35px !important;" in stylesheet
        assert "border-radius: 6px !important;" in stylesheet
        assert "content: var(--description);" not in stylesheet
        assert '--description: "";' in stylesheet
