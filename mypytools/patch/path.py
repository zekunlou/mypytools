"""
Patches for path related issues.
"""

import os


def protect_cwd(func):
    """
    A decorator to protect cwd.
    Useful for dangerous `os.chdir` operations.
    """
    def wrapper(*args, **kwargs):
        """wrapper"""
        cwd = os.getcwd()
        try:
            return func(*args, **kwargs)
        finally:
            """
            The finally clause runs whether or not the try statement produces an exception
            """
            os.chdir(cwd)
    return wrapper


@protect_cwd
def chdir_exec(path, func, *args, **kwargs):
    """
    'cd xxx; exec func; cd -'

    Change directory and execute a function.
    Will go back to the original directory after execution (thanks to the decorator `protect_cwd`)
    """
    os.chdir(path)
    return func(*args, **kwargs)


