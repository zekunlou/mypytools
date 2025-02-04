import numpy


class NPArrWithInfo(numpy.ndarray):
    def __new__(cls, input_array, info=None):
        obj = numpy.asarray(input_array).view(cls)
        obj._info = info
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._info = getattr(obj, "_info", None)

    @property
    def info(self):
        return self._info

    @info.setter
    def info(self, value):
        self._info = value

    def __repr__(self):
        return print_arrinfo(self, do_print=False)

    def __str__(self):
        return print_arrinfo(self, do_print=False)

    def toarr(self):
        """remove the info attribute and return a numpy array"""
        return self.view(numpy.ndarray)


def print_arrinfo(arr: NPArrWithInfo, do_print=True):
    assert isinstance(arr, NPArrWithInfo), "Input array must be an instance of NPArrWithInfo"
    info_str = f"array: {arr.dtype} {arr.shape}\n"

    def _rec_dict2str(d: dict, level: int = 0) -> str:
        _prefix_space = "  " * level
        _str = ""
        for k, v in d.items():
            if isinstance(v, dict):
                _str += f"{_prefix_space}{k}: {_rec_dict2str(v, level + 1)}"
            elif isinstance(v, numpy.ndarray):
                _str += f"{_prefix_space}{k}: {v.dtype} {v.shape}\n"
            elif isinstance(v, int) or isinstance(v, float) or isinstance(v, bool) or isinstance(v, complex):
                _str += f"{_prefix_space}{k}: {type(v).__name__} {v}\n"
            elif isinstance(v, str):
                if len(v) < 10:
                    _str += f"{_prefix_space}{k}: str '{v}'\n"
                else:
                    _str += f"{_prefix_space}{k}: str '{v[:7]}...'\n"
            elif isinstance(v, list) or isinstance(v, tuple) or isinstance(v, set):
                v_str = str(v)
                if len(v_str) < 20:
                    _str += f"{_prefix_space}{k}: {type(v).__name__} {v_str}\n"
                else:
                    _str += f"{_prefix_space}{k}: {type(v).__name__} {v_str[:16]}...\n"
            else:
                _str += f"{_prefix_space}{k}: type {type(v)}\n"
        return _str

    info_str += _rec_dict2str(arr.info, level=1)
    if do_print:
        print(info_str)
    else:
        return info_str
