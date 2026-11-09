"""Unit tests for the NWB form styling helpers (label humanizing + HTML chrome builders)."""

import panel as pn

from guppy.frontend import nwb_form_style as style


class TestHumanize:
    def test_title_cases_each_word(self):
        assert style.humanize("optical_fiber") == "Optical Fiber"
        assert style.humanize("numerical_aperture") == "Numerical Aperture"

    def test_moves_a_trailing_unit_into_parentheses(self):
        assert style.humanize("excitation_wavelength_in_nm") == "Excitation Wavelength (nm)"
        assert style.humanize("core_diameter_in_um") == "Core Diameter (µm)"

    def test_acronym_as_first_word(self):
        # The acronym map wins over capitalization.
        assert style.humanize("ap_in_mm") == "AP (mm)"
        assert style.humanize("id") == "ID"


class TestChromeBuilders:
    def test_page_header_is_the_title_alone(self):
        pane = style.page_header("NWB Metadata")
        assert isinstance(pane, pn.pane.Markdown)
        assert "NWB Metadata" in pane.object

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


class TestTypeScale:
    def test_field_labels_are_14px_regular(self):
        assert "  font-size: 14px;\n  font-weight: 400;" in style.INPUT_STYLESHEET

    def test_page_header_title_is_16px_semibold(self):
        html = style.page_header("NWB Metadata").object

        assert "font-size:16px;font-weight:600;" in html

    def test_section_label_is_14px_semibold_without_uppercase(self):
        html = style.section_label("Session").object

        assert "font-size:14px;font-weight:600;" in html
        assert "uppercase" not in html

    def test_help_note_is_12px_in_the_secondary_gray(self):
        assert "font-size:12px;color:#6C757D;" in style.help_note("Load any saved YAML.").object
