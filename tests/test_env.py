"""
Tests for mypytools.env module.
"""

import os
import sys

import pytest

from mypytools import env


class TestFindRoot:
    """Tests for the find_root function."""

    def test_default_behavior(self, clean_env_vars):
        """Test find_root with default arguments."""
        root = env.find_root()
        assert os.path.isdir(root)
        assert os.path.exists(os.path.join(root, "pyproject.toml"))

    def test_search_from_directory(self, clean_env_vars):
        """Test find_root with search_from parameter as directory."""
        test_dir = os.path.dirname(__file__)
        root = env.find_root(search_from=test_dir)
        assert os.path.isdir(root)
        assert os.path.exists(os.path.join(root, "pyproject.toml"))

    def test_search_from_file(self, clean_env_vars, tmp_work_dir):
        """Test find_root with search_from parameter as file path."""
        # Create a temporary file in a subdirectory
        sub_dir = tmp_work_dir / "subdir"
        sub_dir.mkdir()
        temp_file = sub_dir / "test_file.py"
        temp_file.write_text("# test file")

        # Create project indicator in parent directory
        (tmp_work_dir / "pyproject.toml").write_text("[project]\nname='test'")
        root = env.find_root(search_from=str(temp_file))
        assert root == str(tmp_work_dir)

    @pytest.mark.parametrize(
        "indicator", ["pyproject.toml", ["pyproject.toml"], ["setup.py", "pyproject.toml"], (".git", "pyproject.toml")]
    )
    def test_indicators(self, indicator, clean_env_vars):
        """Test find_root with different indicator formats."""
        root = env.find_root(indicator=indicator)
        assert os.path.isdir(root)
        # Should find at least one of the indicators
        if isinstance(indicator, str):
            assert os.path.exists(os.path.join(root, indicator))
        else:
            assert any(os.path.exists(os.path.join(root, ind)) for ind in indicator)

    def test_project_root_env_var_true(self, clean_env_vars):
        """Test that PROJECT_ROOT environment variable is set when enabled."""
        root = env.find_root(project_root_env_var=True)
        assert os.environ.get("PROJECT_ROOT") == root

    def test_project_root_env_var_false(self, clean_env_vars):
        """Test that PROJECT_ROOT environment variable is not set when disabled."""
        env.find_root(project_root_env_var=False)
        assert "PROJECT_ROOT" not in os.environ

    def test_pythonpath_modification(self, clean_env_vars):
        """Test that sys.path is modified when pythonpath=True."""
        original_path = sys.path.copy()
        root = env.find_root(pythonpath=True)
        assert root in sys.path
        assert sys.path.index(root) == 0  # Should be inserted at beginning
        # Clean up
        sys.path = original_path

    def test_cwd_modification(self, clean_env_vars, tmp_work_dir):
        """Test that current working directory is changed when cwd=True."""
        # Create indicator in temp directory
        (tmp_work_dir / "pyproject.toml").write_text("[project]\nname='test'")
        original_cwd = os.getcwd()
        try:
            root = env.find_root(search_from=str(tmp_work_dir), cwd=True)
            assert os.getcwd() == root == str(tmp_work_dir)
        finally:
            os.chdir(original_cwd)

    @pytest.mark.parametrize("max_depth", [1, 2, 5])
    def test_max_depth_parameter(self, max_depth, clean_env_vars):
        """Test find_root with different max_depth values."""
        root = env.find_root(max_depth=max_depth)
        assert os.path.isdir(root)

    def test_verbose_output(self, clean_env_vars, capsys):
        """Test verbose output is printed when verbose=True."""
        env.find_root(verbose=True, max_depth=1)
        captured = capsys.readouterr()
        assert "depth=" in captured.out
        assert "searching" in captured.out

    def test_max_depth_exceeded(self, tmp_work_dir, clean_env_vars):
        """Test RecursionError when max_depth is exceeded."""
        # Create deep directory structure without any indicators
        deep_dir = tmp_work_dir
        for i in range(5):
            deep_dir = deep_dir / f"level_{i}"
            deep_dir.mkdir()

        with pytest.raises(RecursionError, match="Max depth of 1 exceeded"):
            env.find_root(search_from=str(deep_dir), max_depth=1, indicator="nonexistent_file")


class TestFindRootErrorHandling:
    """Test error handling in find_root function."""

    def test_invalid_search_from_path(self, clean_env_vars):
        """Test FileNotFoundError for invalid search_from path."""
        with pytest.raises(FileNotFoundError, match="invalid_path does not exist"):
            env.find_root(search_from="invalid_path")

    def test_invalid_search_from_type(self, clean_env_vars):
        """Test TypeError for invalid search_from type."""
        with pytest.raises(TypeError, match="start_path must be a string or None"):
            env.find_root(search_from=123)

    @pytest.mark.parametrize("invalid_indicator", [1, 123.456, {"key": "value"}, None])
    def test_invalid_indicator_type(self, invalid_indicator, clean_env_vars):
        """Test TypeError for invalid indicator types."""
        if invalid_indicator == {"key": "value"}:
            # Dictionary is technically iterable, but causes RecursionError due to invalid paths
            with pytest.raises(RecursionError, match="Max depth of 3 exceeded"):
                env.find_root(indicator=invalid_indicator)
        else:
            with pytest.raises(TypeError, match="indicator must be a string or Iterable"):
                env.find_root(indicator=invalid_indicator)

    def test_invalid_max_depth_type(self, clean_env_vars):
        """Test TypeError for invalid max_depth type."""
        with pytest.raises(TypeError):
            env.find_root(max_depth="invalid")

    def test_mixed_indicator_types_in_iterable(self, clean_env_vars):
        """Test that mixed types in indicator iterable raise AssertionError."""
        with pytest.raises(AssertionError):
            env.find_root(indicator=["valid_string", 123])


class TestFindRootIntegration:
    """Integration tests for find_root function."""

    def test_real_project_structure(self, clean_env_vars):
        """Test find_root on the actual project structure."""
        # This should find the mypytools project root
        root = env.find_root()
        expected_files = ["pyproject.toml", "mypytools", "tests"]

        for expected in expected_files:
            assert os.path.exists(os.path.join(root, expected)), f"Expected {expected} not found in project root {root}"
        # Verify it's the correct project
        pyproject_path = os.path.join(root, "pyproject.toml")
        with open(pyproject_path) as f:
            content = f.read()
            assert "mypytools" in content

    def test_multiple_indicators_priority(self, tmp_work_dir, clean_env_vars):
        """Test that find_root works correctly with multiple indicators."""
        # Create both indicators
        (tmp_work_dir / ".git").mkdir()
        (tmp_work_dir / "setup.py").write_text("# setup file")
        root = env.find_root(search_from=str(tmp_work_dir), indicator=[".git", "setup.py"])
        assert root == str(tmp_work_dir)
