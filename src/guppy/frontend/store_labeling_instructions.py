import logging
import os

import holoviews as hv
import numpy as np
import panel as pn

pn.extension()

logger = logging.getLogger(__name__)


def _validate_timestamp_configuration(*, timestamp_column_name: str, time_unit: str) -> None:
    """Raise ValueError if either NPM timestamp selection is empty.

    Kept as a standalone helper so the validation logic can be unit-tested
    without constructing the Panel widgets that feed it.
    """
    missing_fields = []
    if not timestamp_column_name:
        missing_fields.append("'Select which timestamps to use'")
    if not time_unit:
        missing_fields.append("'Select timestamps unit'")
    if missing_fields:
        message = (
            f"NPM timestamp configuration incomplete: {', '.join(missing_fields)} "
            "must be selected before continuing."
        )
        logger.error(message)
        raise ValueError(message)


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
    ts_unit_needs : list of bool, optional
        One entry per NPM data file; ``True`` when the file has multiple
        timestamp columns and requires column/unit selection.
    col_names_ts : list of str, optional
        Timestamp-column options offered in the column selectors.
    """

    def __init__(
        self,
        folder_path: str,
        *,
        channel_previews: dict[str, dict[str, np.ndarray]],
        multiple_event_ttls: list[bool] | None = None,
        ts_unit_needs: list[bool] | None = None,
        col_names_ts: list[str] | None = None,
    ) -> None:
        super().__init__(folder_path=folder_path)
        self.multiple_event_ttls = multiple_event_ttls
        self.ts_unit_needs = ts_unit_needs
        self.col_names_ts = col_names_ts

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

                                            """
        )

        # Per-file configuration widgets, keyed by file index. Only files that
        # actually need input get a widget, mirroring the old per-file dialogs.
        self.split_event_checkboxes: dict[int, pn.widgets.Checkbox] = {}
        self.timestamp_column_selects: dict[int, pn.widgets.Select] = {}
        self.time_unit_selects: dict[int, pn.widgets.Select] = {}
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

            for file_index, needs_unit in enumerate(ts_unit_needs):
                if needs_unit:
                    column_select = pn.widgets.Select(
                        name=f"File {file_index}: select which timestamps to use",
                        options=list(col_names_ts),
                        value=col_names_ts[0],
                        width=550,
                    )
                    unit_select = pn.widgets.Select(
                        name=f"File {file_index}: select timestamps unit",
                        options=["", "seconds", "milliseconds", "microseconds"],
                        value="",
                        width=550,
                    )
                    self.timestamp_column_selects[file_index] = column_select
                    self.time_unit_selects[file_index] = unit_select
                    config_form.extend([column_select, unit_select])

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

    def get_timestamp_configuration(self) -> tuple[list[str], list[str | None]]:
        """Return the per-file timestamp units and column names.

        Files that do not need disambiguation default to ``"seconds"`` and a
        ``None`` column name. Files that do need it are validated and raise
        ``ValueError`` if the column or unit is unset.

        Returns
        -------
        ts_units : list of str
            Time unit for each NPM data file.
        npm_timestamp_column_names : list of str or None
            Selected timestamp column for each file, or ``None`` when no
            selection was required.
        """
        ts_units, npm_timestamp_column_names = [], []
        for file_index, needs_unit in enumerate(self.ts_unit_needs):
            if not needs_unit:
                ts_units.append("seconds")
                npm_timestamp_column_names.append(None)
                continue
            column_name = self.timestamp_column_selects[file_index].value
            time_unit = self.time_unit_selects[file_index].value
            _validate_timestamp_configuration(timestamp_column_name=column_name, time_unit=time_unit)
            ts_units.append(time_unit)
            npm_timestamp_column_names.append(column_name)
        return ts_units, npm_timestamp_column_names

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
