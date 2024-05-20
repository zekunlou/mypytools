import os
import sys
from typing import Iterable, Union


def find_root(
    search_from: str = None,
    indicator: Union[Iterable[str], str] = (
        ".git",
        ".env",
        ".project-root",
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
    ),
    project_root_env_var: bool = True,
    pythonpath: bool = False,
    cwd: bool = False,
    max_depth: int = 3,
    verbose: bool = False,
):
    """
    Find the project root directory.

    Args:
        search_from (str, optional): The path to search from. Defaults to None.
        indicator (Union[Iterable[str], str], optional): The indicator(s) to search for. \
            Has default values (please see source code).
        project_root_env_var (bool, optional): Whether to check the PROJECT_ROOT environment variable. \
            Defaults to True.
        pythonpath (bool, optional): Whether to check the PYTHONPATH environment variable. \
            Defaults to False.
        cwd (bool, optional): Whether to search from the current working directory. Defaults to False.
        max_depth (int, optional): The maximum depth to search. Defaults to 3.
        verbose (bool, optional): Whether to print the search path. Defaults to False.

    Returns:
        str: The path to the project root directory.

    Raises:
        ValueError: If the project root directory cannot be found.
    """

    if search_from is None:
        search_from = os.getcwd()
    elif isinstance(search_from, str):
        if os.path.exists(search_from):
            if not os.path.isdir(search_from):
                search_from = os.path.dirname(search_from)
            else:
                pass
        else:
            raise FileNotFoundError(f"{search_from} does not exist.")
    else:
        raise TypeError(f"start_path must be a string or None, not {type(search_from)}")
    if isinstance(indicator, str):
        indicator = [indicator]
    elif isinstance(indicator, Iterable):
        assert all(isinstance(i, str) for i in indicator)
    else:
        raise TypeError(f"indicator must be a string or Iterable, not {type(indicator)}")

    def _rec_find(
        path: str,
        identifier: Union[Iterable[str], str],
        depth: int,
        max_depth: int,
    ):
        if depth > max_depth:
            raise RecursionError(f"Max depth of {max_depth} exceeded. Please check your project structure.")
        if verbose:
            print(f"{depth=}, searching {path=}")
        for i in identifier:
            if os.path.exists(os.path.join(path, i)):
                return path
        return _rec_find(os.path.dirname(path), identifier, depth + 1, max_depth)

    root = _rec_find(search_from, indicator, 0, max_depth)

    if project_root_env_var:
        os.environ["PROJECT_ROOT"] = root

    if pythonpath:
        sys.path.insert(0, root)

    if cwd:
        os.chdir(root)

    return root
