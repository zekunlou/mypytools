import time
from dataclasses import dataclass

import numpy


@dataclass
class EVDResult:
    # say the original matrix is (N, N)
    eigenvalues: numpy.ndarray  # (N,)
    eigenvectors: numpy.ndarray  # (N, N)
    time: float = 0.0
    target_path: str = None  # where you can find the original matrix

    def __repr__(self):
        return (
            f"EVDResult(eigenvalues={self.eigenvalues.shape}, eigenvectors={self.eigenvectors.shape}, "
            f"time={self.time:.3e} sec, target_path={self.target_path})"
        )

    def save(self, fpath: str):
        numpy.savez(
            fpath,
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors,
            time=self.time,
            target_path=self.target_path,
        )

    @classmethod
    def load(cls, fpath: str):
        data = numpy.load(fpath)
        try:
            data["target_path"]
        except ValueError:
            return cls(
                eigenvalues=data["eigenvalues"],
                eigenvectors=data["eigenvectors"],
                time=data["time"],
            )
        else:
            return cls(
                eigenvalues=data["eigenvalues"],
                eigenvectors=data["eigenvectors"],
                time=data["time"],
                target_path=data["target_path"],
            )

    @classmethod
    def compute(cls, matrix: numpy.ndarray, target_path: str = None, hermitian: bool = False):
        start_time = time.time()
        if hermitian:
            eigenvalues, eigenvectors = numpy.linalg.eigh(matrix)
        else:
            eigenvalues, eigenvectors = numpy.linalg.eig(matrix)
        end_time = time.time()
        return cls(eigenvalues, eigenvectors, end_time - start_time, target_path)
