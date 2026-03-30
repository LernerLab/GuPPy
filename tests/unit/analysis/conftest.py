import numpy as np
import pytest


@pytest.fixture
def uniform_signal():
    """1000-point sine wave plus small noise at 100 Hz for 10 seconds."""
    rng = np.random.default_rng(seed=42)
    time = np.linspace(0, 10, 1000)
    return np.sin(2 * np.pi * 1.0 * time) + 0.05 * rng.standard_normal(1000)
