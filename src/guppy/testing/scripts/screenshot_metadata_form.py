"""Dev tool: render the Step 6 metadata form and screenshot it.

Serves :func:`guppy.orchestration.metadata.build_metadata_template` on a free port,
drives a headless Chromium to the page, optionally expands every collapsible card,
and writes a full-page PNG. Used to iterate on the form's visual design; it is not a
test (no assertions). Requires playwright + chromium in the active environment.

Run from the project root:
    python -m guppy.testing.scripts.screenshot_metadata_form /tmp/metadata.png \\
        [--expand] [--example tests/data/fiber_photometry_metadata_example.yaml]

    --example PATH   prefill the form from a metadata YAML on disk
    --expand         click every collapsible card open before capturing
"""

import socket
import sys
import time

import panel as pn
from playwright.sync_api import sync_playwright

from guppy.frontend.frontend_utils import scanPortsAndFind
from guppy.orchestration.metadata import build_metadata_template
from guppy.utils.nwb_metadata import (
    Channel,
    build_metadata_dict,
    load_yaml,
    parse_metadata_dict,
)


def main() -> None:
    """Serve the metadata form, capture a full-page screenshot to the path in argv[1]."""
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/metadata.png"
    expand = "--expand" in sys.argv
    example_path = sys.argv[sys.argv.index("--example") + 1] if "--example" in sys.argv else None

    pn.extension(notifications=True)
    channels = [Channel("dms", "control", "Dv1A"), Channel("dms", "signal", "Dv2A")]

    metadata = {}
    if example_path:
        example = load_yaml(example_path)
        devices, _rows, scalars = parse_metadata_dict(metadata=example, channels=channels)
        rows = [
            {
                "excitation_wavelength_in_nm": 405.0,
                "emission_wavelength_in_nm": 525.0,
                "indicator": "dms_green_fluorophore",
                "optical_fiber": "optical_fiber",
                "excitation_source": "excitation_source_isosbestic_control",
                "photodetector": "photodetector",
            },
            {
                "excitation_wavelength_in_nm": 465.0,
                "emission_wavelength_in_nm": 525.0,
                "indicator": "dms_green_fluorophore",
                "optical_fiber": "optical_fiber",
                "excitation_source": "excitation_source_calcium_signal",
                "photodetector": "photodetector",
            },
        ]
        scalars = {
            **scalars,
            "session_description": "RI30 photometry session",
            "lab": "Lerner Lab",
            "institution": "Northwestern University",
            "subject_id": "63_207",
            "sex": "M",
            "species": "Mus musculus",
            "experimenter": ["Adkisson, Paul"],
        }
        metadata = build_metadata_dict(devices=devices, channel_rows=rows, scalars=scalars, channels=channels)

    template = build_metadata_template(
        session_label="Photo_63 (1)",
        channels=channels,
        metadata=metadata,
        metadata_yaml_path="/tmp/nwb_metadata.yaml",
    )
    port = scanPortsAndFind(start_port=5000, end_port=5200)
    pn.serve(template, port=port, show=False, threaded=True)
    for _ in range(80):
        try:
            socket.create_connection(("localhost", port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.05)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 1400})
        page.goto(f"http://localhost:{port}", wait_until="networkidle")
        time.sleep(1.5)
        if expand:
            # Several passes: opening an outer card is what makes the cards nested inside it
            # visible, and only a visible header can be clicked.
            for _ in range(3):
                cards = page.locator("div.card-header")
                for index in range(cards.count()):
                    card = cards.nth(index)
                    if card.is_visible():
                        card.click()
                time.sleep(0.5)
            time.sleep(1.0)
        page.screenshot(path=out_path, full_page=True)
        browser.close()
    print(f"wrote {out_path}")
    pn.state.kill_all_servers()


if __name__ == "__main__":
    main()
