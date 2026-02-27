import glob
import logging
import os

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn

# hv.extension()
pn.extension()

logger = logging.getLogger(__name__)


class StorenamesInstructions:
    def __init__(self, folder_path):
        # instructions about how to save the storeslist file
        self.mark_down = pn.pane.Markdown(
            """


                    ### Instructions to follow :

                    - Check Storenames to repeat checkbox and see instructions in “Github Wiki” for duplicating storenames.
                    Otherwise do not check the Storenames to repeat checkbox.<br>
                    - Select storenames from list and click “Select Storenames” to populate area below.<br>
                    - Enter names for storenames, in order, using the following naming convention:<br>
                        Isosbestic = “control_region” (ex: Dv1A= control_DMS)<br>
                        Signal= “signal_region” (ex: Dv2A= signal_DMS)<br>
                        TTLs can be named using any convention (ex: PrtR = RewardedPortEntries) but should be kept consistent for later group analysis

                    ```
                    {"storenames": ["Dv1A", "Dv2A",
                                    "Dv3B", "Dv4B",
                                    "LNRW", "LNnR",
                                    "PrtN", "PrtR",
                                    "RNPS"],
                    "names_for_storenames": ["control_DMS", "signal_DMS",
                                            "control_DLS", "signal_DLS",
                                            "RewardedNosepoke", "UnrewardedNosepoke",
                                            "UnrewardedPort", "RewardedPort",
                                            "InactiveNosepoke"]}
                    ```
                    - If user has saved storenames before, clicking "Select Storenames" button will pop up a dialog box
                    showing previously used names for storenames. Select names for storenames by checking a checkbox and
                    click on "Show" to populate the text area in the Storenames GUI. Close the dialog box.

                    - Select “create new” or “overwrite” to generate a new storenames list or replace a previous one
                    - Click Save

                    """,
            width=550,
        )

        self.widget = pn.Column("# " + os.path.basename(folder_path), self.mark_down)


class StorenamesInstructionsNPM(StorenamesInstructions):
    def __init__(self, folder_path):
        super().__init__(folder_path=folder_path)
        path_chev = glob.glob(os.path.join(folder_path, "*chev*"))
        path_chod = glob.glob(os.path.join(folder_path, "*chod*"))
        path_chpr = glob.glob(os.path.join(folder_path, "*chpr*"))
        combine_paths = path_chev + path_chod + path_chpr
        self.d = dict()
        for i in range(len(combine_paths)):
            basename = (os.path.basename(combine_paths[i])).split(".")[0]
            df = pd.read_csv(combine_paths[i])
            self.d[basename] = {"x": np.array(df["timestamps"]), "y": np.array(df["data"])}
        keys = list(self.d.keys())
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
                                        - When you give your storenames, name the channels appropriately.
                                        For example, “chev1” might be “signal_A” and
                                        “chod1” might be “control_A” (or vice versa).

                                            """
        )
        self.plot_select = pn.widgets.Select(
            name="Select channel to see correspondings channels", options=keys, value=keys[0]
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

    def _make_plot(self, plot_key):
        return hv.Curve((self.d[plot_key]["x"], self.d[plot_key]["y"])).opts(width=550)

    def _on_plot_select_change(self, event):
        self.plot_pane.object = self._make_plot(event.new)
