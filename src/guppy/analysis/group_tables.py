"""Collect per-run summary tables (tonic epoch means, covariate correlations) across a group's members."""

import pandas as pd


def stack_member_tables(*, member_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack every member's table into one, keyed by member label.

    Parameters
    ----------
    member_tables : dict of str to pd.DataFrame
        Each member run's table, keyed by the member's label. Every table must share the
        same index names and columns.

    Returns
    -------
    pd.DataFrame
        The member tables concatenated in order, with ``member`` prepended as the outermost
        index level.
    """
    return pd.concat(member_tables, names=["member"])


def summarize_member_tables(*, member_tables: dict[str, pd.DataFrame], value_columns: list[str]) -> pd.DataFrame:
    """Summarize each value across members as its mean, standard error and member count.

    Members with a NaN value are left out of that value's mean, standard error and count.

    Parameters
    ----------
    member_tables : dict of str to pd.DataFrame
        Each member run's table, keyed by the member's label, all sharing one index.
    value_columns : list of str
        Columns to summarize.

    Returns
    -------
    pd.DataFrame
        One row per index entry, in the first member's order, with ``<column>_mean``,
        ``<column>_sem`` and ``<column>_n`` for every column in ``value_columns``. The
        standard error is the sample standard deviation over the square root of the count,
        so it is NaN when fewer than two members have a value.
    """
    first_table = next(iter(member_tables.values()))
    summary = pd.DataFrame(index=first_table.index)
    for column in value_columns:
        values = pd.concat([table[column] for table in member_tables.values()], axis=1).loc[first_table.index]
        summary[column + "_mean"] = values.mean(axis=1)
        summary[column + "_sem"] = values.sem(axis=1)
        summary[column + "_n"] = values.count(axis=1)
    return summary
