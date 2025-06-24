import time
from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy


@dataclass
class SVDResult:
    # say the original matrix is (M, N), and K = max(M, N)
    U: numpy.ndarray  # (M, M)
    S: numpy.ndarray  # (K,)
    Vh: numpy.ndarray  # (N, N)
    time: float = 0.0
    target_path: str = None  # where you can find the original matrix
    # rebuilt_cond: float = None  # condition number of the rebuilt matrix

    def __repr__(self):
        return (
            f"SVDResult(U={self.U.shape}, S={self.S.shape}, Vh={self.Vh.shape}, "
            f"time={self.time:.3e} sec, target_path={self.target_path})"
        )

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
        try:
            data["target_path"]
        except ValueError:
            return cls(
                U=data["U"],
                S=data["S"],
                Vh=data["Vh"],
                time=data["time"],
                # target_path=data["target_path"],
            )
        else:
            return cls(
                U=data["U"],
                S=data["S"],
                Vh=data["Vh"],
                time=data["time"],
                target_path=data["target_path"],
            )

    @classmethod
    def compute(cls, matrix: numpy.ndarray, hermitian: bool, target_path: str = None):
        start_time = time.time()
        U, S, Vh = numpy.linalg.svd(matrix, full_matrices=True, compute_uv=True, hermitian=hermitian)
        end_time = time.time()
        return cls(U, S, Vh, end_time - start_time, target_path)


def SVD_truncation(
    svd_result: SVDResult,
    eps: Optional[float] = None,
    return_rebuilt: bool = True,
    return_pinv: bool = True,
) -> tuple[numpy.ndarray, tuple[int, int]]:
    """
    https://math.stackexchange.com/questions/19948/pseudoinverse-matrix-and-svd

    Equation: A^+ = (A^T A)^{-1} A^T = V S^{-1} U^T
    while S is filtered by a threshold epsilon

    Return:
    - the pseudo-inverse matrix
    - how many singular values are kept
    - how many singular values are there in total
    """

    if eps is None:
        use_indices = numpy.arange(svd_result.S.size)
    else:
        assert isinstance(eps, float) and eps > 0.0
        use_indices = numpy.where(svd_result.S > eps)[0]  # NOQA
    U_use = svd_result.U[:, use_indices]
    S_use = svd_result.S[use_indices]
    Vh_use = svd_result.Vh[use_indices, :]
    mat_rebuilt = (U_use * S_use) @ Vh_use if return_rebuilt else None
    mat_pinv = (Vh_use.T * S_use ** (-1)) @ U_use.T if return_pinv else None
    return {
        "mat_pinv": mat_pinv,
        "mat_rebuilt": mat_rebuilt,
        "num_sv_used": len(use_indices),  # number of singular values used
        "num_sv_total": len(svd_result.S),  # number of singular values in total
    }


def SVD_smooth(
    svd_result: SVDResult,
    eps: float = 1e-10,
    method: Literal["Tikhonov",] = "Tikhonov",
    ignore_small_filter: Optional[float] = None,
    return_rebuilt: bool = True,
    return_pinv: bool = True,
) -> tuple[numpy.ndarray, tuple[int, int]]:
    """
    https://www.imm.dtu.dk/~pcha/HNO/chap6.pdf

    Equation: A^+ = V f S^+ U^T
    where f is a filter function applied to the singular

    Return:
    - the pseudo-inverse matrix
    - how many singular values are kept (if ignore_small_filter is not None)
    - how many singular values are there in total
    """
    assert method in [
        "Tikhonov",
    ]

    def _filter(_x, _eps):
        if method == "Tikhonov":
            return _x**2 / (_x**2 + _eps**2)
        else:
            raise ValueError(f"Unknown method: {method}")

    f_values = _filter(svd_result.S, eps)

    if ignore_small_filter is None:
        use_indices = numpy.arange(svd_result.S.size)
    else:
        use_indices = numpy.where(f_values > ignore_small_filter)[0]
    U_use = svd_result.U[:, use_indices]
    S_use = svd_result.S[use_indices]
    Vh_use = svd_result.Vh[use_indices, :]
    mat_rebuilt = (U_use * f_values[use_indices] * S_use) @ Vh_use if return_rebuilt else None
    mat_pinv = (Vh_use.T * f_values[use_indices] * S_use ** (-1)) @ U_use.T if return_pinv else None
    return {
        "mat_pinv": mat_pinv,
        "mat_rebuilt": mat_rebuilt,
        "num_sv_g_eps": len(numpy.where(f_values > eps)[0]),  # number of singular values greater than eps
        "num_sv_used": len(use_indices),  # number of singular values used
        "num_sv_total": len(svd_result.S),  # number of singular values in total
    }


def cal_L_curve(
    svd_result: SVDResult,
    b: numpy.ndarray,
    eps_array: numpy.ndarray,
    pseudo_inverse_method: Union[Literal["truncation"], Literal["smooth"]],
    solution_norm_method: Union[Literal["x2"], Literal["xAx"]],
):
    """
    Compute the L-curve for A x = b, return ||A x - b|| by ||x|| or (x^T A x)^0.5
    """
    assert eps_array.ndim == 1
    assert eps_array.dtype == float
    assert pseudo_inverse_method in ["truncation", "smooth"]
    assert solution_norm_method in ["x2", "xAx"]

    residual_norms = []
    solution_norms = []
    for eps in eps_array:
        if pseudo_inverse_method == "truncation":
            A_pinv = SVD_truncation(svd_result, eps=eps)
        elif pseudo_inverse_method == "smooth":
            A_pinv = SVD_smooth(svd_result, eps=eps)
        else:
            raise ValueError(f"Unknown method: {pseudo_inverse_method}")
        x = A_pinv["mat_pinv"] @ b
        residual_norms.append(numpy.linalg.norm(A_pinv["mat_rebuilt"] @ x - b))
        if solution_norm_method == "x2":
            solution_norms.append(numpy.linalg.norm(x))
        elif solution_norm_method == "xAx":
            solution_norms.append((x.T @ A_pinv["mat_rebuilt"] @ x) ** 0.5)

    return residual_norms, solution_norms
