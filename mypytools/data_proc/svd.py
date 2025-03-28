from dataclasses import dataclass

import numpy


@dataclass
class SVDResult:
    U: numpy.ndarray
    S: numpy.ndarray
    Vh: numpy.ndarray
    time: float = 0.0
    target_path: str = None  # where you can find the original matrix
    # rebuilt_cond: float = None  # condition number of the rebuilt matrix

    def save(self, fpath: str):
        numpy.savez(
            fpath,
            U=self.U,
            S=self.S,
            Vh=self.Vh,
            time=self.time,
            target_path=self.target_path,
        )

    @classmethod
    def load(cls, fpath: str):
        data = numpy.load(fpath)
        return cls(
            U=data["U"],
            S=data["S"],
            Vh=data["Vh"],
            time=data["time"],
            target_path=data["target_path"],
        )
