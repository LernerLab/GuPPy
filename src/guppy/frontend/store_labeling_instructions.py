import logging
import os

import holoviews as hv
import numpy as np
import panel as pn

from guppy.extractors.npm_recording_extractor import (
    DEFAULT_TIME_UNIT,
    TIME_UNIT_DIVISORS,
)

pn.extension()

logger = logging.getLogger(__name__)

TIME_UNIT_OPTIONS = list(TIME_UNIT_DIVISORS)


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
                    - For each store, choose a **Type** and provide a label:<br>
                        **signal** — enter the recording-site name (e.g. `DMS`). Any name is allowed,
                        including underscores (e.g. `left_hemisphere`).<br>
                        **control** — choose, from the **Control for** dropdown, which signal this
                        control belongs to. The pair name is taken from that signal, so you enter it
                        only once and a control can never be mismatched to the wrong recording site.<br>
                        **event TTLs** — enter an event name (e.g. `RewardedPortEntries`). Keep event
                        names consistent across sessions for later group analysis.<br>
                    - If user has saved stores before, clicking "Select Stores" pre-fills each store with its
                    previously used Type and label.
                    - Select “create new” or “overwrite” to generate a new store_ids list or replace a previous one
                    - Click Save

                    """,
            width=550,
        )

        self.widget = pn.Column("# " + os.path.basename(folder_path), self.mark_down)


class StoreLabelingInstructionsNPM(StoreLabelingInstructions):
    """Label Stores instructions panel extended with NPM-specific configuration and previews.

    Renders per-file widgets so the user can choose whether to split multiple
    behavior TTLs into separate files and which timestamp column/unit to use,
    plus a "Confirm NPM configuration" button whose click handler is wired by
    the orchestrator. After confirmation the orchestrator decomposes the NPM
    session and calls :meth:`set_channel_previews` to render the channel traces.

    Parameters
    ----------
    folder_path : str
        Absolute path to the NPM session directory; its basename is shown as a
        heading above the instructions.
    channel_previews : dict
        Maps each chev/chod/chpr channel name to a dict with ``"x"`` (timestamps)
        and ``"y"`` (data) arrays to plot. Pass an empty dict to start with no
        preview (populated later via :meth:`set_channel_previews`).
    multiple_event_ttls : list of bool, optional
        One entry per NPM data file; ``True`` when the file encodes multiple TTL
        types and a split-events checkbox should be shown. When ``None`` the
        interactive configuration form is not built.
    timestamp_column_options : list of str, optional
        Timestamp columns the session's data files offer. A column selector is
        shown only when there are two or more to choose between.
    """

    def __init__(
        self,
        folder_path: str,
        *,
        channel_previews: dict[str, dict[str, np.ndarray]],
        multiple_event_ttls: list[bool] | None = None,
        timestamp_column_options: list[str] | None = None,
    ) -> None:
        super().__init__(folder_path=folder_path)
        self.multiple_event_ttls = multiple_event_ttls
        self.timestamp_column_options = timestamp_column_options

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
                                        - Label the channels using the Type dropdowns. For example,
                                        mark “chev1” as **signal** and name it “A”, then mark “chod1”
                                        as **control** and set its **Control for** to “chev1” (or vice
                                        versa).

                                            """,
            width=550,
        )

        # Split-events is asked per file, keyed by file index, and only for the files
        # that encode more than one TTL type. The timestamp widgets below are per session.
        self.split_event_checkboxes: dict[int, pn.widgets.Checkbox] = {}
        self.timestamp_column_select: pn.widgets.Select | None = None
        self.time_unit_select: pn.widgets.Select | None = None
        self.confirm_button: pn.widgets.Button | None = None
        config_form = pn.Column()

        if multiple_event_ttls is not None:
            for file_index, has_multiple in enumerate(multiple_event_ttls):
                if has_multiple:
                    checkbox = pn.widgets.Checkbox(
                        name=f"File {file_index}: create multiple files for each behavior type?",
                        value=False,
                        width=550,
                    )
                    self.split_event_checkboxes[file_index] = checkbox
                    config_form.append(checkbox)

            # A session is recorded on one clock, so the timestamp column and unit are
            # asked once for the whole folder. The column is only ambiguous when the
            # files offer more than one to choose between; the unit always is.
            if timestamp_column_options is not None and len(timestamp_column_options) > 1:
                self.timestamp_column_select = pn.widgets.Select(
                    name="Select which timestamps to use",
                    options=list(timestamp_column_options),
                    width=550,
                )
                config_form.append(self.timestamp_column_select)

            self.time_unit_select = pn.widgets.Select(
                name="Select the unit of the timestamps",
                options=TIME_UNIT_OPTIONS,
                value=DEFAULT_TIME_UNIT,
                width=550,
            )
            config_form.append(self.time_unit_select)

            self.confirm_button = pn.widgets.Button(name="Confirm NPM configuration", width=550)
            config_form.append(self.confirm_button)

        # Preview area is filled by set_channel_previews (immediately if previews
        # were supplied, otherwise after the user confirms the configuration).
        self.channel_preview_arrays: dict[str, dict[str, np.ndarray]] = {}
        self.plot_select: pn.widgets.Select | None = None
        self.plot_pane: pn.pane.HoloViews | None = None
        self.plot_area = pn.Column()

        self.widget = pn.Column(
            "# " + os.path.basename(folder_path),
            self.mark_down,
            self.mark_down_np,
            config_form,
            self.plot_area,
        )

        if channel_previews:
            self.set_channel_previews(channel_previews=channel_previews)

    def get_npm_split_events(self) -> list[bool]:
        """Return, per NPM data file, whether to split multiple behavior TTLs.

        Files that do not encode multiple TTL types are always ``False``;
        the rest reflect their split-events checkbox.

        Returns
        -------
        list of bool
            One entry per NPM data file.
        """
        return [
            bool(self.split_event_checkboxes[file_index].value) if has_multiple else False
            for file_index, has_multiple in enumerate(self.multiple_event_ttls)
        ]

    def get_timestamp_configuration(self) -> tuple[str, str | None]:
        """Return the session's timestamp unit and column.

        Returns
        -------
        npm_time_unit : str
            Unit the session's timestamps are recorded in.
        npm_timestamp_column_name : str or None
            Selected timestamp column, or ``None`` when the files offer only one.
        """
        npm_timestamp_column_name = (
            self.timestamp_column_select.value if self.timestamp_column_select is not None else None
        )
        return self.time_unit_select.value, npm_timestamp_column_name

    def set_channel_previews(self, *, channel_previews: dict[str, dict[str, np.ndarray]]) -> None:
        """Render (or re-render) the channel selector and preview plot.

        Parameters
        ----------
        channel_previews : dict
            Maps each chev/chod/chpr channel name to a dict with ``"x"`` and
            ``"y"`` arrays to plot.
        """
        self.channel_preview_arrays = {
            name: {"x": np.asarray(preview["x"]), "y": np.asarray(preview["y"])}
            for name, preview in channel_previews.items()
        }
        channel_names = list(self.channel_preview_arrays.keys())
        self.plot_select = pn.widgets.Select(
            name="Select channel to see correspondings channels", options=channel_names, value=channel_names[0]
        )
        self.plot_pane = pn.pane.HoloViews(self._make_plot(self.plot_select.value), width=550)
        self.plot_select.param.watch(self._on_plot_select_change, "value")
        self.plot_area.objects = [self.plot_select, self.plot_pane]

    def _make_plot(self, plot_key: str) -> hv.Curve:
        preview = self.channel_preview_arrays[plot_key]
        return hv.Curve((preview["x"], preview["y"])).opts(width=550)

    def _on_plot_select_change(self, event: object) -> None:
        self.plot_pane.object = self._make_plot(event.new)
