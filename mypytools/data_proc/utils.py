import numpy
import pandas


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
        def rec_repr(d: dict, offset=""):
            return "\n".join(
                [
                    (
                        offset + f"{k}: {repr(v)}"
                        if not isinstance(v, dict)
                        else offset + f"{k}:\n{rec_repr(v, offset+'  ')}"
                    )
                    for k, v in d.items()
                ]
            )

        return rec_repr(self._mydict, offset="")

    def __getattr__(self, name):
        assert name in self._mydict.keys(), f"{name} not in {self._mydict.keys()}"
        value = self._mydict[name]
        if isinstance(value, dict):
            return AttrDict(value)
        else:
            return value

    def to_dict(
        self,
    ):
        """get the original dict"""
        return self._mydict


def format_e(x):
    return "0" if x == 0.0 else "1e" + str(int(numpy.log10(x)))


def dict2str(d):
    return "__".join([f"{k}:{v}" for k, v in d.items()])


def str2dict(s):
    return dict([kv.split(":") for kv in s.split("__")])


def update_df_info_to_col(df: pandas.DataFrame, drop_info: bool = True) -> pandas.DataFrame:
    """inplace operation for df!"""
    assert "info" in df.columns
    info_list = [str2dict(i) for i in df["info"].tolist()]
    info_keys = set(k for d in info_list for k in d.keys())
    for k in info_keys:
        df[k] = [d.get(k, None) for d in info_list]
    if drop_info:
        df.drop(columns=["info"], inplace=True)
    return df
