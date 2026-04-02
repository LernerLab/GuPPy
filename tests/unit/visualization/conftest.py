import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_all_figures():
    yield
    plt.close("all")
