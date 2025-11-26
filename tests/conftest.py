"""Pytest configuration for Orbbec tests."""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-hardware", action="store_true", help="run hardware-dependent tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "hardware: mark test as hardware-dependent")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-hardware"):
        return
    skip_hw = pytest.mark.skip(reason="--run-hardware not set")
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip_hw)


@pytest.fixture(scope="session", autouse=True)
def log_env():
    try:
        import pyorbbecsdk  # type: ignore

        print("[conftest] pyorbbecsdk ok", pyorbbecsdk.__file__)
    except Exception as exc:  # pragma: no cover - diagnostic only
        print("[conftest] pyorbbecsdk import failed", exc)
    import sys, os

    print("[conftest] sys.path", sys.path)
    print("[conftest] LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH"))
    print("[conftest] sys.version", sys.version)
