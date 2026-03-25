import numpy as np
import pytest


@pytest.fixture
def uniform_signal():
    """1000-point sine wave plus small noise at 100 Hz for 10 seconds."""
    rng = np.random.default_rng(seed=42)
    time = np.linspace(0, 10, 1000)
    return np.sin(2 * np.pi * 1.0 * time) + 0.05 * rng.standard_normal(1000)


@pytest.fixture
def uniform_timestamps():
    """1000 evenly-spaced timestamps from 0 to 10 seconds at 100 Hz."""
    return np.linspace(0, 10, 1000)


@pytest.fixture
def two_coord_windows():
    """Two non-overlapping good-data windows: [1.0, 4.0] and [6.0, 9.0]."""
    return np.array([[1.0, 4.0], [6.0, 9.0]])
