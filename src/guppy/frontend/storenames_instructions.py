import logging

import panel as pn

# hv.extension()
pn.extension()

logger = logging.getLogger(__name__)


class StorenamesInstructions:
    def __init__(self):
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
