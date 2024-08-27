class AttrDict:
    """Access dict keys as attributes

    The attr trick only works for nested dicts.
    One just needs to wrap the dict in an AttrDict object.

    Example:
    ```python
    d = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    d = AttrDict(d)
    print(d.a)  # 1
    print(d.b.c)  # 2
    print(d.b.d.e)  # 3
    print(d.to_dict()["b"]["d"]["e"])  # 3
    ```
    """
    def __init__(self, d: dict):
        self._mydict = d

    def __repr__(self):
        def rec_repr(d: dict, offset=''):
            return '\n'.join([
                offset + f"{k}: {repr(v)}"
                if not isinstance(v, dict)
                else offset + f"{k}:\n{rec_repr(v, offset+'  ')}"
                for k, v in d.items()
            ])
        return rec_repr(self._mydict, offset='')

    def __getattr__(self, name):
        assert name in self._mydict.keys(), f"{name} not in {self._mydict.keys()}"
        value = self._mydict[name]
        if isinstance(value, dict):
            return AttrDict(value)
        else:
            return value

    def to_dict(self,):
        """get the original dict"""
        return self._mydict
