"""
Pytest configuration and shared fixtures for mypytools tests.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def tmp_work_dir(tmp_path):
    """Create a temporary working directory and change to it."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(original_cwd)


@pytest.fixture
def sample_numpy_array():
    """Generate a sample numpy array for testing."""
    np.random.seed(42)  # For reproducibility
    return np.random.rand(10, 10)


@pytest.fixture
def sample_matrix_data():
    """Generate sample matrix data for SVD and linear algebra tests."""
    np.random.seed(42)
    return {
        "small_matrix": np.random.rand(5, 5),
        "rectangular_matrix": np.random.rand(10, 5),
        "symmetric_matrix": np.eye(5) + 0.1 * np.random.rand(5, 5),
    }


@pytest.fixture
def mock_aims_output(test_data_dir):
    """Path to mock FHI-aims output files for testing."""
    aims_data_dir = test_data_dir / "tag_output_rs_matrices_h5_250806"
    if aims_data_dir.exists():
        return aims_data_dir
    else:
        pytest.skip("FHI-aims test data not available")


@pytest.fixture
def sample_coordinates():
    """Generate sample atomic coordinates for structure tests."""
    return np.array(
        [
            [0.0, 0.0, 0.0],  # Atom 1
            [1.0, 0.0, 0.0],  # Atom 2
            [0.0, 1.0, 0.0],  # Atom 3
            [0.0, 0.0, 1.0],  # Atom 4
        ]
    )


@pytest.fixture
def sample_xyz_content():
    """Sample XYZ file content as string."""
    return """4
Sample molecule
H    0.000000    0.000000    0.000000
H    1.000000    0.000000    0.000000
O    0.000000    1.000000    0.000000
O    0.000000    0.000000    1.000000
"""


@pytest.fixture(scope="function")
def clean_env_vars():
    """Clean environment variables before and after test."""
    # Store original values
    original_env = {}
    test_vars = ["PROJECT_ROOT", "PYTHONPATH"]

    for var in test_vars:
        if var in os.environ:
            original_env[var] = os.environ[var]
            del os.environ[var]

    yield

    # Restore original values
    for var in test_vars:
        if var in original_env:
            os.environ[var] = original_env[var]
        elif var in os.environ:
            del os.environ[var]


def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "mpi: mark test as requiring MPI")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add slow marker to tests with "slow" in name
        if "slow" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)

        # Add integration marker to tests with "integration" in name
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)

        # Add gpu marker to tests in gpu module or with "gpu" in name
        if "gpu" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)

        # Add mpi marker to tests in mpi module or with "mpi" in name
        if "mpi" in item.nodeid.lower():
            item.add_marker(pytest.mark.mpi)
