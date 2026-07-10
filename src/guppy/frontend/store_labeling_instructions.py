import logging
import os

import holoviews as hv
import numpy as np
import panel as pn

pn.extension()

logger = logging.getLogger(__name__)


class StoreLabelingInstructions:
    """Panel widget displaying store_ids-configuration instructions for a session folder.

    Parameters
    ----------
    folder_path : str
        Absolute path to the session directory; its basename is shown as a
        heading above the instructions.
    """

    def __init__(self, folder_path: str) -> None:
        # instructions about how to save the storeslist file
        self.mark_down = pn.pane.Markdown(
            """


                    ### Instructions to follow :

                    - Check Stores to repeat checkbox and see instructions in “Github Wiki” for duplicating stores.
                    Otherwise do not check the Stores to repeat checkbox.<br>
                    - Select stores from list and click “Select Stores” to populate area below.<br>
                    - Enter names for store_ids, in order, using the following naming convention:<br>
                        Isosbestic = “control_region” (ex: Dv1A= control_DMS)<br>
                        Signal= “signal_region” (ex: Dv2A= signal_DMS)<br>
                        TTLs can be named using any convention (ex: PrtR = RewardedPortEntries) but should be kept consistent for later group analysis

                    ```
                    {"store_ids": ["Dv1A", "Dv2A",
                                    "Dv3B", "Dv4B",
                                    "LNRW", "LNnR",
                                    "PrtN", "PrtR",
                                    "RNPS"],
                    "store_labels": ["control_DMS", "signal_DMS",
                                            "control_DLS", "signal_DLS",
                                            "RewardedNosepoke", "UnrewardedNosepoke",
                                            "UnrewardedPort", "RewardedPort",
                                            "InactiveNosepoke"]}
                    ```
                    - If user has saved stores before, clicking "Select Stores" button will pop up a dialog box
                    showing previously used names for store_ids. Select names for store_ids by checking a checkbox and
                    click on "Show" to populate the text area in the Label Stores GUI. Close the dialog box.

                    - Select “create new” or “overwrite” to generate a new store_ids list or replace a previous one
                    - Click Save

                    """,
            width=550,
        )

        self.widget = pn.Column("# " + os.path.basename(folder_path), self.mark_down)


class StoreLabelingInstructionsNPM(StoreLabelingInstructions):
    """Label Stores instructions panel extended with NPM-specific channel preview plots.

    Adds a channel selector and a live HoloViews curve so the user can inspect
    each NPM channel before assigning it a signal or control role. The decomposed
    channel traces are computed upstream (by the orchestrator, via
    ``NpmRecordingExtractor.decompose``) and passed in; this widget only renders
    them.

    Parameters
    ----------
    folder_path : str
        Absolute path to the NPM session directory; its basename is shown as a
        heading above the instructions.
    channel_previews : dict
        Maps each chev/chod/chpr channel name to a dict with ``"x"`` (timestamps)
        and ``"y"`` (data) arrays to plot.
    """

    def __init__(self, folder_path: str, *, channel_previews: dict[str, dict[str, np.ndarray]]) -> None:
        super().__init__(folder_path=folder_path)
        self.channel_preview_arrays = {
            name: {"x": np.asarray(preview["x"]), "y": np.asarray(preview["y"])}
            for name, preview in channel_previews.items()
        }
        channel_names = list(self.channel_preview_arrays.keys())
        self.mark_down_np = pn.pane.Markdown(
            """
                                        ### Extra Instructions to follow when using Neurophotometrics data :
                                        - Guppy will take the NPM data, which has interleaved frames
                                        from the signal and control channels, and divide it out into
                                        separate channels for each site you recordded.
                                        However, since NPM does not automatically annotate which
                                        frames belong to the signal channel and which belong to the
                                        control channel, the user must specify this for GuPPy.
                                        - Each of your recording sites will have a channel
                                        named “chod” and a channel named “chev”
                                        - View the plots below and, for each site,
                                        determine whether the “chev” or “chod” channel is signal or control
                                        - When you give your store_ids, name the channels appropriately.
                                        For example, “chev1” might be “signal_A” and
                                        “chod1” might be “control_A” (or vice versa).

                                            """
        )
        self.plot_select = pn.widgets.Select(
            name="Select channel to see correspondings channels", options=channel_names, value=channel_names[0]
        )
        self.plot_pane = pn.pane.HoloViews(self._make_plot(self.plot_select.value), width=550)
        self.plot_select.param.watch(self._on_plot_select_change, "value")

        self.widget = pn.Column(
            "# " + os.path.basename(folder_path),
            self.mark_down,
            self.mark_down_np,
            self.plot_select,
            self.plot_pane,
        )

    def _make_plot(self, plot_key: str) -> hv.Curve:
        preview = self.channel_preview_arrays[plot_key]
        return hv.Curve((preview["x"], preview["y"])).opts(width=550)

    def _on_plot_select_change(self, event: object) -> None:
        self.plot_pane.object = self._make_plot(event.new)
