"""Read and write ``storesList.csv``, the store id ↔ store label mapping for a run.

Step 1 writes one ``storesList.csv`` into every run folder and the rest of the
pipeline reads it back to learn which stores exist and what they are called. The
file is a headerless two-row CSV: row 0 holds the raw store ids as they appear in
the acquisition format (``Dv1A``, ``PrtA``, ...) and row 1 holds the GuPPy store
labels the user assigned to them (``control_DMS``, ``signal_DMS``, ...). Column
``i`` of one row pairs with column ``i`` of the other, and that column order is
itself meaningful — the NWB converter walks recording sites in it.
"""

import os

import numpy as np

STORES_LIST_FILENAME = "storesList.csv"
COMBINED_STORES_LIST_FILENAME = "combine_storesList.csv"


def read_stores_list(*, run_folder: str, filename: str = STORES_LIST_FILENAME) -> np.ndarray:
    """Read a run folder's store mapping.

    Parameters
    ----------
    run_folder : str
        Directory holding the store-mapping CSV.
    filename : str, optional
        Name of the CSV within ``run_folder``. Defaults to ``storesList.csv``.

    Returns
    -------
    np.ndarray
        String array of shape ``(2, n_stores)``: row 0 store ids, row 1 store
        labels. A single-store file still comes back 2-D.
    """
    return np.genfromtxt(os.path.join(run_folder, filename), dtype="str", delimiter=",").reshape(2, -1)


def write_stores_list(*, run_folder: str, store_array: np.ndarray, filename: str = STORES_LIST_FILENAME) -> None:
    """Write a store mapping into a run folder.

    Parameters
    ----------
    run_folder : str
        Directory to write the store-mapping CSV into.
    store_array : np.ndarray
        String array of shape ``(2, n_stores)``: row 0 store ids, row 1 store
        labels.
    filename : str, optional
        Name of the CSV within ``run_folder``. Defaults to ``storesList.csv``.
    """
    np.savetxt(os.path.join(run_folder, filename), store_array, delimiter=",", fmt="%s")
