"""Pytest configuration and shared fixtures."""

import pytest
import torch


def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if CUDA not available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture
def device():
    """Get appropriate device for tests."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    batch_size = 4
    seq_len = 128
    state_dim = 256

    return {
        "input_ids": torch.randn(batch_size, seq_len, state_dim),
        "labels": torch.randn(batch_size, seq_len, state_dim),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }
