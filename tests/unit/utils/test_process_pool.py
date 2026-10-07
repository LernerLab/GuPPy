import os

import pytest

from guppy.utils.process_pool import run_starmap


def process_id_and_sum(first: int, second: int) -> tuple[int, int]:
    return os.getpid(), first + second


class TestRunStarmap:
    def test_a_single_process_runs_in_place(self):
        results = run_starmap(function=process_id_and_sum, arguments=[(1, 2), (3, 4), (5, 6)], process_count=1)

        assert results == [(os.getpid(), 3), (os.getpid(), 7), (os.getpid(), 11)]

    def test_an_exception_reaches_the_caller(self):
        with pytest.raises(TypeError):
            run_starmap(function=process_id_and_sum, arguments=[(1, None)], process_count=1)

    @pytest.mark.parallel
    def test_several_processes_run_in_spawned_workers(self):
        results = run_starmap(function=process_id_and_sum, arguments=[(1, 2), (3, 4), (5, 6)], process_count=2)

        assert [total for _, total in results] == [3, 7, 11]
        assert os.getpid() not in {process_id for process_id, _ in results}
