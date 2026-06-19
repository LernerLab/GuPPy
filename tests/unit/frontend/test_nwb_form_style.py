"""Unit tests for the NWB form styling helpers (label humanizing + HTML chrome builders)."""

import panel as pn

from guppy.frontend import nwb_form_style as style


class TestHumanize:
    def test_capitalizes_first_word_only(self):
        assert style.humanize("optical_fiber") == "Optical fiber"
        assert style.humanize("numerical_aperture") == "Numerical aperture"

    def test_preserves_units_and_acronyms_mid_label(self):
        assert style.humanize("excitation_wavelength_in_nm") == "Excitation wavelength in nm"

    def test_acronym_as_first_word(self):
        # The acronym map wins over first-word capitalization.
        assert style.humanize("ap_in_mm") == "AP in mm"
        assert style.humanize("id") == "ID"


class TestChromeBuilders:
    def test_page_header_includes_title_and_subtitle(self):
        pane = style.page_header("NWB metadata", "Photo (run1)")
        assert isinstance(pane, pn.pane.Markdown)
        assert "NWB metadata" in pane.object
        assert "Photo (run1)" in pane.object

    def test_section_label_with_and_without_note(self):
        with_note = style.section_label("Session", "last name first")
        assert "Session" in with_note.object
        assert "last name first" in with_note.object
        # The no-note path renders the label without the trailing note span.
        without_note = style.section_label("Subject")
        assert "Subject" in without_note.object

    def test_subgroup_help_and_intro_notes_carry_text(self):
        assert "Hardware" in style.subgroup_label("Hardware").object
        assert "be careful" in style.help_note("be careful").object
        assert "welcome" in style.intro_note("welcome").object

    def test_channel_role_chip_for_signal_and_control(self):
        signal = style.channel_role_chip("signal", "Dv2A")
        assert "signal" in signal.object
        assert "Dv2A" in signal.object
        # The control role takes the alternate (muted) color branch.
        control = style.channel_role_chip("control", "Dv1A")
        assert "control" in control.object
        assert "Dv1A" in control.object
