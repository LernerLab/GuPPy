import multiprocessing as mp
from collections.abc import Callable, Iterable
from itertools import starmap


def run_starmap(*, function: Callable[..., object], arguments: Iterable[tuple], process_count: int) -> list[object]:
    """Call ``function`` on each argument tuple, in a spawned process pool or in this process.

    Parameters
    ----------
    function : callable
        Module-level function to call; it must be picklable when ``process_count`` is above 1.
    arguments : iterable of tuple
        One tuple of positional arguments per call.
    process_count : int
        Number of worker processes. At 1 or below the calls run serially in this process.

    Returns
    -------
    list
        The return values, in the order of ``arguments``.
    """
    if process_count <= 1:
        # A spawned worker re-imports GuPPy and its dependencies, which takes seconds; with a
        # single worker there is no parallelism to pay that for.
        return list(starmap(function, arguments))

    # Pinned rather than inherited: callers run on a background thread of the Panel server
    # process, and forking a process that has other live threads can leave a lock they held
    # (logging, HDF5) permanently locked in the child.
    spawn_context = mp.get_context("spawn")
    with spawn_context.Pool(process_count) as pool:
        results = pool.starmap(function, arguments)
        # Close and join before leaving the block. The context manager's __exit__ calls terminate(),
        # which signals every worker and then blocks in waitpid() until it is gone; a worker that is
        # slow to exit never gets there and the parent waits forever. starmap has already returned,
        # so there is nothing to abort -- close() lets each worker exit on its own.
        pool.close()
        pool.join()
    return results
