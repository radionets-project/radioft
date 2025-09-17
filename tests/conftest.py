import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--run-large", action="store_true", default=False, help="run large memory tests"
    )
    parser.addoption(
        "--max-memory-gb",
        action="store",
        default=4,
        type=int,
        help="maximum memory to use for tests in GB",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "large: mark test as large memory usage")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-large"):
        skip_large = pytest.mark.skip(reason="need --run-large option to run")
        for item in items:
            if "large" in item.keywords:
                item.add_marker(skip_large)


@pytest.fixture(scope="session")
def max_memory_gb(request):
    return request.config.getoption("--max-memory-gb")


@pytest.fixture(scope="session")
def device():
    """Return the device to use for tests"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
