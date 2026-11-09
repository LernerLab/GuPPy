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

    def test_writes_every_unit_after_in_as_its_symbol(self):
        assert style.humanize("pitch_in_deg") == "Pitch (deg)"
        assert style.humanize("volume_in_uL") == "Volume (µL)"
        assert style.humanize("exposure_time_in_s") == "Exposure Time (s)"
        assert style.humanize("intensity_in_W_per_m2") == "Intensity (W/m²)"
        assert style.humanize("titer_in_vg_per_ml") == "Titer (vg/mL)"

    def test_hyphenates_cut_on_and_cut_off(self):
        assert style.humanize("cut_on_wavelength_in_nm") == "Cut-On Wavelength (nm)"
        assert style.humanize("cut_off_wavelength_in_nm") == "Cut-Off Wavelength (nm)"

    def test_acronym_as_first_word(self):
        # The acronym map wins over capitalization.
        assert style.humanize("ap_in_mm") == "AP (mm)"
        assert style.humanize("id") == "ID"


class TestChromeBuilders:
    def test_section_box_is_a_white_homepage_section(self):
        field = pn.widgets.TextInput(name="Lab")
        section = style.section_box("Session", field)

        assert isinstance(section, pn.WidgetBox)
        assert section.styles == {
            "background": "White",
            "border": "1px solid #C8CCD0",
            "border-radius": "6px",
            "margin-bottom": "12px",
        }
        assert section[0].object == "### Session"
        assert section[1] is field

    def test_section_box_puts_its_note_under_the_heading_in_small_text(self):
        section = style.section_box("Experimenter", note="Last name first.")

        assert section[1].object == "Last name first."
        assert section[1].css_classes == ["small"]

    def test_subgroup_help_and_intro_notes_carry_text(self):
        assert "Hardware" in style.subgroup_label("Hardware").object
        assert "be careful" in style.help_note("be careful").object
        assert "welcome" in style.intro_note("welcome").object

    def test_channel_role_chip_names_the_role_in_its_plot_color(self):
        signal = style.channel_role_chip("signal", "Dv2A").object
        assert "color:#3A9E5C;background:#3A9E5C1A;" in signal
        assert ">Signal</span>" in signal
        assert "Dv2A" in signal

        control = style.channel_role_chip("control", "Dv1A").object
        assert "color:#1F6FD1;background:#1F6FD11A;" in control
        assert ">Control</span>" in control
        assert "Dv1A" in control


class TestTypeScale:
    def test_help_note_is_12px_in_the_secondary_gray(self):
        assert "font-size:12px;color:#6C757D;" in style.help_note("Load any saved YAML.").object
