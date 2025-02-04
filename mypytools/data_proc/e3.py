from typing import Union

import numpy
from scipy.linalg import expm


class Irreps(tuple):
    """Handle irreducible representation arrays, like slices, multiplicities, etc."""

    def __new__(cls, irreps: Union[str, list[int], tuple[int]]) -> "Irreps":
        """Create an Irreps object

        Args:
            irreps (Union[str, list[int], tuple[int]]): irreps info
                - str, e.g. `1x0+2x1+3x2+3x3+2x4+1x5`
                    - multiplicities and l values joined by `x`
                - tuple[tuple[int]], e.g. ((1, 0), (2, 1), (3, 2), (3, 3), (2, 4), (1, 5),)
                    - each tuple is (multiplicity, l)
                - tuple[int], e.g. (0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5,)
                    - list of l values, the multiplicities are calculated automatically

        Notes:
            The internal representation is a tuple of tuples, each tuple is (multiplicity, l).
            e.g. ((1, 0), (2, 1), (3, 2), (3, 3), (2, 4), (1, 5),)

        Returns:
            Irreps object
        """
        if isinstance(irreps, str):
            irreps_info_split = tuple(sec.strip() for sec in irreps.split("+") if len(sec) > 0)  # ("1x0", "2x1", ...)
            mul_l_tuple = tuple(  # ((1, 0), (2, 1), ...)
                tuple(int(i.strip()) for i in sec.split("x")) for sec in irreps_info_split
            )
            return super().__new__(cls, mul_l_tuple)
        elif isinstance(irreps, list) or isinstance(irreps, tuple):
            if len(irreps) == 0:
                return super().__new__(cls, ())
            elif isinstance(irreps[0], tuple) or isinstance(irreps[0], list):
                assert all(
                    all(isinstance(i, int) for i in mul_l) and len(mul_l) == 2 and mul_l[0] >= 0 and mul_l[1] >= 0
                    for mul_l in irreps
                ), ValueError(f"Invalid irreps_info: {irreps}")
                return super().__new__(cls, tuple(tuple(mul_l) for mul_l in irreps))
            elif isinstance(irreps[0], int):
                assert all(isinstance(i, int) and i >= 0 for i in irreps), ValueError(
                    f"Invalid irreps format: {irreps}"
                )
                this_l_cnt, this_l = 1, irreps[0]
                mul_l_list: list[tuple[int]] = []
                for l_num in irreps[1:]:
                    if l_num == this_l:
                        this_l_cnt += 1
                    else:
                        mul_l_list.append((this_l_cnt, this_l))
                        this_l_cnt, this_l = 1, l_num
                mul_l_list.append((this_l_cnt, this_l))
                # print(mul_l_list)
                return super().__new__(cls, tuple(mul_l_list))
            else:
                raise ValueError(f"Invalid irreps format: {irreps}")
        else:
            raise ValueError(f"Invalid irreps format: {irreps}")

    @property
    def dim(self):
        """total dimension / length by magnetic quantum number"""
        return sum(mul * (2 * l_num + 1) for mul, l_num in self)

    @property
    def num_irreps(self):
        """number of irreps, the sum of multiplicities of each l"""
        return sum(mul for mul, _ in self)

    @property
    def ls(self) -> list[int]:
        """list of l values in the irreps"""
        return tuple(l_num for mul, l_num in self for _ in range(mul))

    @property
    def lmax(self) -> int:
        """maximum l in the irreps"""
        return max(tuple(l_num for _, l_num in self))

    def __repr__(self):
        return "+".join(f"{mul}x{l_num}" for mul, l_num in self)

    def __add__(self, other: "Irreps") -> "Irreps":
        return Irreps(super().__add__(other))

    def slices(self) -> list[slice]:
        """return all the slices for each l"""
        if hasattr(self, "_slices"):
            return self._slices
        else:
            self._slices = []
            ls = self.ls
            l_m_nums = tuple(2 * l_num + 1 for l_num in ls)
            pointer = 0
            for m_num in l_m_nums:
                self._slices.append(slice(pointer, pointer + m_num))
                pointer += m_num
            assert pointer == self.dim
            self._slices = tuple(self._slices)
        return self._slices

    def slices_l(self, l_num: int) -> list[slice]:
        """return all the slices for a specific l"""
        return tuple(sl for _l, sl in zip(self.ls, self.slices()) if l_num == _l)

    def simplify(self) -> "Irreps":
        """sort by l, and combine the same l"""
        uniq_ls = tuple(set(self.ls))
        mul_ls = tuple((self.ls.count(l_num), l_num) for l_num in uniq_ls)
        return Irreps(mul_ls)

    def sort(self) -> "Irreps":
        """sort by l, return the sorted Irreps and the permutation"""
        raise NotImplementedError

    def D_from_angles(self, alpha: float, beta: float, gamma: float) -> numpy.ndarray:
        f"""calculate the Wigner D matrix for the given rotation angles

        docstring: {wigner_D.__doc__}

        Rotation operation:
            gamma around Y, beta around X, alpha around Y.
        """
        Ds = [(mul, wigner_D(l_num, alpha, beta, gamma)) for mul, l_num in self]
        for _, D in Ds:
            assert D.dtype == numpy.float64, ValueError(f"Invalid dtype: {D.dtype}")
        Ds = [D for mul, D in Ds for _ in range(mul)]
        # concatenate along the diagonal
        Ds_len = numpy.array([D.shape[0] for D in Ds])
        Ds_cumsum = numpy.hstack([0, numpy.cumsum(Ds_len)])
        D = numpy.zeros((Ds_cumsum[-1], Ds_cumsum[-1]), dtype=numpy.float64)
        for this_D, this_start in zip(Ds, Ds_cumsum[:-1]):
            this_len = this_D.shape[0]
            D[this_start : this_start + this_len, this_start : this_start + this_len] = this_D
        return D


"""
below is adapted from https://github.com/e3nn/e3nn/blob/main/e3nn/o3/_wigner.py#L11
"""


def su2_generators(j: int) -> numpy.ndarray:
    m = numpy.arange(-j, j)
    raising = numpy.diag(-numpy.sqrt(j * (j + 1) - m * (m + 1)), k=-1)

    m = numpy.arange(-j + 1, j + 1)
    lowering = numpy.diag(numpy.sqrt(j * (j + 1) - m * (m - 1)), k=1)

    m = numpy.arange(-j, j + 1)
    return numpy.stack(
        [
            0.5 * (raising + lowering),  # x (usually)
            numpy.diag(1j * m),  # z (usually)
            -0.5j * (raising - lowering),  # -y (usually)
        ],
        axis=0,
    )


def change_basis_real_to_complex(l: int) -> numpy.ndarray:
    # https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    q = numpy.zeros((2 * l + 1, 2 * l + 1), dtype=numpy.complex128)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = 1 / 2**0.5
        q[l + m, l - abs(m)] = -1j / 2**0.5
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1) ** m / 2**0.5
        q[l + m, l - abs(m)] = 1j * (-1) ** m / 2**0.5
    q = (-1j) ** l * q  # Added factor of 1j**l to make the Clebsch-Gordan coefficients real

    return q


def so3_generators(l) -> numpy.ndarray:
    X = su2_generators(l)
    Q = change_basis_real_to_complex(l)
    X = numpy.conj(Q.T) @ X @ Q
    assert numpy.all(numpy.abs(numpy.imag(X)) < 1e-5)
    return numpy.real(X)


def wigner_D(l: int, alpha: float, beta: float, gamma: float) -> numpy.ndarray:
    """Rotation matrix in the Wigner D matrix representation in SO(3).
    Rotation operation: gamma around Y, beta around X, alpha around Y.

    It satisfies the following properties:

    * :math:`D(\text{identity rotation}) = \text{identity matrix}`
    * :math:`D(R_1 \\circ R_2) = D(R_1) \\circ D(R_2)`
    * :math:`D(R^{-1}) = D(R)^{-1} = D(R)^T`
    * :math:`D(\text{rotation around Y axis})` has some property that allows us to use FFT in `ToS2Grid`
    """
    alpha, beta, gamma = map(lambda x: x % (2 * numpy.pi), (alpha, beta, gamma))
    X = so3_generators(l)
    return expm(alpha * X[1]) @ expm(beta * X[0]) @ expm(gamma * X[1])
