"""
Test configuration.
"""
import pytest


@pytest.fixture
def anyio_backend():
    """Set anyio backend for async tests."""
    return 'asyncio'
