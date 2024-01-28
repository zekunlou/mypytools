"""
test env.py
"""

import os
import sys

import pytest

from mypytools import env


def test_find_root():
    """
    Test find_root().
    """
    indicator = "pyproject.toml"

    # Test with default arguments
    root = env.find_root()
    assert os.path.isdir(root)

    # Test with search_from
    root = env.find_root(search_from=os.path.dirname(__file__))
    assert os.path.isdir(root)

    # Test with indicator
    root = env.find_root(indicator=indicator)
    assert os.path.isdir(root)
    assert os.path.exists(os.path.join(root, indicator))

    # Test with indicator as Iterable
    root = env.find_root(indicator=[indicator])
    assert os.path.isdir(root)
    assert os.path.exists(os.path.join(root, indicator))

    # Test with project_root_env_var
    root = env.find_root(project_root_env_var=False)
    assert os.path.isdir(root)
    assert os.path.exists(os.path.join(root, indicator))

    # Test with pythonpath
    root = env.find_root(pythonpath=True)
    assert os.path.isdir(root)
    assert os.path.exists(os.path.join(root, indicator))
    assert os.path.join(root) in sys.path

    # Test with max_depth
    root = env.find_root(max_depth=1)
    assert os.path.isdir(root)
    assert os.path.exists(os.path.join(root, indicator))

    # Test with verbose
    root = env.find_root(verbose=True)
    assert os.path.isdir(root)
    assert os.path.exists(os.path.join(root, indicator))

    # Test with invalid search_from
    with pytest.raises(FileNotFoundError):
        env.find_root(search_from="invalid_path")

    # Test with invalid indicator
    with pytest.raises(TypeError):
        env.find_root(indicator=1)

    # # Test with invalid project_root_env_var
    # with pytest.raises(TypeError):
    #     env.find_root(project_root_env_var=1)

    # # Test with invalid pythonpath
    # with pytest.raises(TypeError):
    #     env.find_root(pythonpath=1)

    # Test with invalid max_depth
    with pytest.raises(TypeError):
        env.find_root(max_depth="invalid")

    # # Test with invalid verbose
    # with pytest.raises(TypeError):
    #     env.find_root(verbose="invalid")
