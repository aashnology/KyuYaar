import pytest

from data_loader import load_data
from toolkit import Toolkit


@pytest.fixture(scope="session")
def data():
    return load_data()


@pytest.fixture(scope="session")
def orders(data):
    return data[0]


@pytest.fixture(scope="session")
def marketing(data):
    return data[1]


@pytest.fixture(scope="session")
def toolkit(orders, marketing):
    return Toolkit(orders, marketing)
