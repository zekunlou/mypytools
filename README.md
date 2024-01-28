# README

This repo is all about reusable python code blocks by Zekun Lou.

## Tips

- When using jupyter, please consider add `autoreload` to reload modules automatically.
    - For details, see [here](https://ipython.org/ipython-doc/3/config/extensions/autoreload.html).
```python
%load_ext autoreload
%autoreload 1  # Reload all modules imported with %aimport every time before executing the Python code typed.
%aimport [my_module]
```

