from typing import Literal, Union

import numpy


def ridge_regression(
    A: numpy.ndarray,
    b: numpy.ndarray,
    regularizer_method: Union[Literal["x2"], Literal["xAx"]],
):
    """
    A x = b
    Equation:
    minimize ||A x - b||^2 + \lambda ||x||^2
    where \lambda is the regularizer
    Solution
    x = (A^T A + \lambda I)^{-1} A^T b
    minimize ||A x - b||^2 + \lambda x^T A x
    Solution
    x = (A^T A + \lambda A)^{-1} A^T b
    """

    assert regularizer_method in ["x2", "xAx"]
    mat = A.T @ A + numpy.eye(A.shape[1]) if regularizer_method == "x2" else A.T @ A + numpy.eye(A.shape[1]) * A
    x = numpy.linalg.solve(mat, A.T @ b)
    return x
