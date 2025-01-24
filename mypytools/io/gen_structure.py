import logging
from typing import Dict, Literal, Tuple, Union

import numpy
from ase.atoms import Atoms
from ase.build import make_supercell
from ase.data import atomic_masses
from ase.symbols import symbols2numbers

_log = logging.getLogger(__name__)


def slide_twisted_bilayer(
    atoms: Atoms,
    n1: int,
    n2: int,
    z_dist: float,
    shake_upper_layer_z: Union[None, float] = None,
    slide_lower_layer_xy: Union[None, float] = None,
    shake_all_atoms: Union[None, float] = None,
    seed: Union[None, int] = None,
):
    """slide the twisted bilayer, no twisting, just want to mimic thermal fluctuation
    Args:
        atoms: Atoms, the twisted bilayer cell
        n1: int, supercell size in the first lattice vector
        n2: int, supercell size in the second lattice vector
        z_dist: float, distance between two layers
        shake_upper_layer_z: float, shake upper layer z coordinate
        slide_lower_layer_xy: float, slide lower layer xy coordinate
        shake_all_atoms: float, shake all atoms
        seed: int, random seed for reproducibility
    """
    if seed is not None:
        numpy.random.seed(seed)
    """make supercell"""
    cell_sup = make_supercell(atoms, [[n1, 0, 0], [0, n2, 0], [0, 0, 1]])
    cell_sup.cell[2, 2] = z_dist if z_dist > 100 else 100
    cell_midz = numpy.mean(cell_sup.positions[:, 2])
    layer_lower_idxes = numpy.where(cell_sup.positions[:, 2] < cell_midz)[0]
    layer_upper_idxes = numpy.where(cell_sup.positions[:, 2] >= cell_midz)[0]
    layer_lower = cell_sup[layer_lower_idxes].copy()
    layer_upper = cell_sup[layer_upper_idxes].copy()
    interlayer_z_old = numpy.mean(layer_upper.positions[:, 2]) - numpy.mean(layer_lower.positions[:, 2])
    layer_upper.positions[:, 2] += (z_dist - interlayer_z_old)
    """glide upper layer"""
    glide_vec = numpy.sum(atoms.get_cell()[:2] * numpy.random.rand(2).reshape(2, 1), axis=0)
    layer_upper.positions += glide_vec.reshape(1, 3)
    """glide lower layer"""
    if slide_lower_layer_xy is not None:
        assert isinstance(slide_lower_layer_xy, float)
        glide_vec = numpy.sum(atoms.get_cell()[:2] * slide_lower_layer_xy, axis=0)
        layer_lower.positions += glide_vec.reshape(1, 3)
    """shake upper layer z coordinate, only upper layer"""
    if shake_upper_layer_z is not None:
        assert isinstance(shake_upper_layer_z, float)
        shake_val = shake_upper_layer_z * numpy.random.randn()
        layer_upper.positions[:, 2] += shake_val
    """combine the two layers"""
    atoms_comb = Atoms(
        symbols=layer_lower.get_chemical_symbols() + layer_upper.get_chemical_symbols(),
        positions=numpy.concatenate([layer_lower.positions, layer_upper.positions], axis=0),
        cell=cell_sup.cell,
        pbc=cell_sup.pbc,
    )
    """shake all atoms"""
    if shake_all_atoms is not None:
        assert isinstance(shake_all_atoms, float)
        M = atoms_comb.get_masses()
        shake_val = shake_all_atoms * numpy.random.randn(*atoms_comb.positions.shape) * (M ** (-0.5)).reshape(-1, 1)
        atoms_comb.positions += shake_val
    atoms_comb.wrap()
    """center the cell"""
    cell_z = atoms_comb.cell[2, 2]
    atoms_comb.positions[:, 2] -= numpy.mean(atoms_comb.positions[:, 2]) - cell_z / 2
    return atoms_comb


def gen_sliding_bilayer(
    cell_prim: Atoms,
    n1: int,
    n2: int,
    z_dist: float,
    conj: Union[Literal["AA"], Literal["AB"], Literal["AA'"]],
    shake_upper_layer_z: Union[None, float] = None,
    slide_lower_layer_xy: Union[None, float] = None,
    shake_all_atoms: Union[None, float, dict] = None,
    seed: Union[None, int] = None,
) -> Atoms:
    """WARNING: This function is not well-tested yet.
    Generate a bilayer with sliding upper layer.

    Args:
        cell_prim (Atom): primitive cell
        n1 (int): supercell size in x direction
        n2 (int): supercell size in y direction
        z_dist (float): distance between two layers
        conj (str): "AA" or "AB", "AB" for inverting upper layer along z direction
        shake_upper_layer_z (float): shake upper layer z coordinate.
            This is will be used to scale a normal distributed value.
        slide_lower_layer_xy (float): slide lower layer xy coordinate
        shake_all_atoms (float): shake all atoms. The full scaling factor is ${shake_all_atoms}*M**(-0.5),
            and will be used to scale a normal distributed value.
        seed (int): random seed

    Returns:
        Atoms: bilayer cell

    Examples:
    ```python
    bilayer_cell = gen_sliding_bilayer(
        cell_prim=str_lift,
        n1=4,
        n2=4,
        z_dist=5.664,
        conj="AA",
        shake_all_atoms=1.0,
        shake_upper_layer_z=0.6,
        slide_lower_layer_xy=0.1,
        seed=17,
    )
    visualize_cell_2d(bilayer_cell)
    plt.show()
    plt.scatter(bilayer_cell.positions[:, 0], bilayer_cell.positions[:, 2])
    plt.show()
    ```
    """
    if seed is not None:
        numpy.random.seed(seed)
    """make supercell"""
    cell_sup = make_supercell(cell_prim, [[n1, 0, 0], [0, n2, 0], [0, 0, 1]])
    cell_sup.cell[2, 2] = 6 * z_dist if 6 * z_dist > 100 else 100
    layer_lower = cell_sup.copy()
    layer_upper = cell_sup.copy()
    """move upper layer higher"""
    layer_upper.positions[:, 2] += z_dist
    """glide upper layer"""
    # glide_vec = numpy.random.rand() * cell_prim.get_cell()[0]
    # glide_vec += numpy.random.rand() * cell_prim.get_cell()[1]
    glide_vec = numpy.sum(cell_prim.get_cell()[:2] * numpy.random.rand(2).reshape(2, 1), axis=0)
    layer_upper.positions += glide_vec.reshape(1, 3)
    """glide lower layer"""
    if slide_lower_layer_xy is not None:
        assert isinstance(slide_lower_layer_xy, float)
        glide_vec = numpy.sum(cell_prim.get_cell()[:2] * slide_lower_layer_xy, axis=0)
        layer_lower.positions += glide_vec.reshape(1, 3)
    """deal with conj, operate on upper layer only"""
    if conj == "AA":  # do nothing
        pass
    elif conj == "AB":  # z-inversion
        layer_upper_z_mean = numpy.mean(layer_upper.positions[:, 2])
        layer_upper.positions[:, 2] = 2 * layer_upper_z_mean - layer_upper.positions[:, 2]
    elif conj == "AA'":  # xy-parity, and move back to the original cell
        layer_upper_positions = -layer_upper.positions[:, :2]
        layer_upper_positions += n1 * cell_prim.get_cell()[0, :2] + n2 * cell_prim.get_cell()[1, :2]
        layer_upper.positions[:, :2] = layer_upper_positions
    else:
        raise ValueError(f"conj {conj} not supported")
    """shake upper layer z coordinate, only upper layer"""
    if shake_upper_layer_z is not None:
        assert isinstance(shake_upper_layer_z, float)
        shake_val = shake_upper_layer_z * numpy.random.randn()
        layer_upper.positions[:, 2] += shake_val
    """combine the two layers"""
    cell_comb = Atoms(
        symbols=layer_lower.get_chemical_symbols() + layer_upper.get_chemical_symbols(),
        positions=numpy.concatenate([layer_lower.positions, layer_upper.positions], axis=0),
        cell=cell_sup.cell,
        pbc=cell_sup.pbc,
    )
    """shake all atoms"""
    if shake_all_atoms is None:
        pass
    elif isinstance(shake_all_atoms, float):
        M = cell_comb.get_masses()
        shake_val = shake_all_atoms * numpy.random.randn(*cell_comb.positions.shape) * (M ** (-0.5)).reshape(-1, 1)
        cell_comb.positions += shake_val
    elif isinstance(shake_all_atoms, dict):  # e.g. {"Zr": 2.0, "S": 1.0}
        species_names = set(cell_comb.get_chemical_symbols())
        assert species_names == set(shake_all_atoms.keys())
        species_masses: Dict[str, float] = {s: atomic_masses[symbols2numbers(s)[0]] for s in species_names}
        species_idx = {s: numpy.where(numpy.array(cell_comb.get_chemical_symbols()) == s)[0] for s in species_names}
        shake_val = numpy.zeros_like(cell_comb.positions)
        for s in species_names:
            shake_val[species_idx[s]] = (
                shake_all_atoms[s] * numpy.random.randn(len(species_idx[s]), 3) * (species_masses[s] ** (-0.5))
            )
        cell_comb.positions += shake_val
    else:
        raise ValueError(f"shake_all_atoms {shake_all_atoms} not supported, should be float or dict")
    cell_comb.wrap()
    """center the cell"""
    cell_z = cell_comb.cell[2, 2]
    cell_comb.positions[:, 2] -= numpy.mean(cell_comb.positions[:, 2]) - cell_z / 2
    return cell_comb


def gen_twisted_bilayer(
    cell_prim: Atoms,
    m: int,
    r: int,
    z_dist: float,
    conj: Union[Literal["AA"], Literal["AB"], Literal["AA'"]],
    shake_upper_layer_z: Union[None, float] = None,
    shake_all_atoms: Union[None, float] = None,
    wrap: bool = True,
    seed: Union[None, int] = None,
):
    """generate twisted bilayer, for hexagonal lattice only!

    Args:
        cell_prim (Atoms): primitive cell
        m (int): the first integer, defines the twist angle
        r (int): the second integer, defines the twist angle
        z_dist (float): distance between two layers
        conj (str): "AA", "AB", "AA'", bilayer conjugation relation
        shake_upper_layer_z (float): shake upper layer z coordinate.
            This is will be used to scale a normal distributed value.
        shake_all_atoms (float): shake all atoms. The full scaling factor is ${shake_all_atoms}*M**(-0.5),
            and will be used to scale a normal distributed value.
        seed (int): random seed for reproducibility

    Returns:
        dict: {
            "twist_angle": float, the twist angle in radian,
            "atoms": Atoms, the twisted bilayer cell
        }

    Example:
    ```python
    twisted_bilayer = gen_twisted_bilayer(
        cell_prim=cell_prim,
        m=3,
        r=1,
        z_dist=5.664,
        conj="AA",
        shake_all_atoms=1.0,
        shake_upper_layer_z=0.6,
        seed=17,
    )
    visualize_cell_2d(twisted_bilayer["atoms"])
    plt.show()
    plt.scatter(twisted_bilayer["atoms"].positions[:, 0], twisted_bilayer["atoms"].positions[:, 2])
    plt.show()
    ```
    """
    if seed is not None:
        numpy.random.seed(seed)
    assert numpy.allclose(cell_prim.get_cell()[[2, 2, 0, 1], [0, 1, 2, 2]], 0)
    twist_property = get_twist_property(m, r)
    twist_angle = twist_property["angle"]
    supercell_matrix = numpy.diag([1, 1, 1])
    supercell_matrix[:2, :2] = twist_property["suplat_trans"]
    """ generate the bottom layer """
    supercell0 = make_supercell(cell_prim, supercell_matrix, wrap=wrap)
    supercell0.cell[2, 2] = 6 * z_dist if 6 * z_dist > 100 else 100
    supercell0_point = supercell0.get_cell()[:2, :2].sum(axis=0)
    theta = numpy.arctan(supercell0_point[1] / supercell0_point[0]) / numpy.pi * 180
    supercell0.rotate(-theta, "z", rotate_cell=True)  # align the furthest cell vertex to x-axis
    """ generate the top layer """
    supercell1 = supercell0.copy()
    supercell1.positions[:, 1] *= -1  # invert y-axis
    """deal with conj, operate on upper layer only"""
    if conj == "AA":  # do nothing
        pass
    elif conj == "AB":  # z-inversion
        supercell1_z_mean = numpy.mean(supercell1.positions[:, 2])
        supercell1.positions[:, 2] = 2 * supercell1_z_mean - supercell1.positions[:, 2]
    elif conj == "AA'":
        supercell0_point = supercell1.get_cell()[:2, :2].sum(axis=0)
        supercell1.positions[:, :2] = supercell0_point - supercell1.positions[:, :2]
    """ shake upper layer z coordinate, only upper layer """
    supercell1.positions[:, 2] += z_dist  # lift the top layer
    if shake_upper_layer_z is not None:
        assert numpy.allclose(cell_prim.get_cell()[[2, 2, 0, 1], [0, 1, 2, 2]], 0)
        shake_val = shake_upper_layer_z * numpy.random.randn()
        supercell1.positions[:, 2] += shake_val
    """ combine the two layers, rotate to ipi format """
    supercell_comb = supercell0 + supercell1
    supercell_comb.rotate(30, "z", rotate_cell=True)
    """ shake atoms """
    if shake_all_atoms is not None:
        assert isinstance(shake_all_atoms, float)
        M = supercell_comb.get_masses()
        shake_val = shake_all_atoms * numpy.random.randn(*supercell_comb.positions.shape) * (M ** (-0.5)).reshape(-1, 1)
        supercell_comb.positions += shake_val
    """ center the cell and wrap up """
    supercell_z = supercell_comb.cell[2, 2]
    supercell_comb.positions[:, 2] -= numpy.mean(supercell_comb.positions[:, 2]) - supercell_z / 2
    if wrap:
        supercell_comb.wrap()
    supercell_comb.info["twist_angle"] = twist_angle
    supercell_comb.info["twist_m"] = m
    supercell_comb.info["twist_r"] = r
    return supercell_comb


def get_rot2D_mat(theta: float):
    """2D rotation matrix"""
    return numpy.array(
        [
            [numpy.cos(theta), -numpy.sin(theta)],
            [numpy.sin(theta), numpy.cos(theta)],
        ]
    )


def get_unique_arr(arr: numpy.ndarray, **kwargs) -> numpy.ndarray:
    """Return the unique rows of a (1+n)D array.
    The rows are considered to be unique if they are not close to each other.

    Args:
        arr: (1+n)D numpy array, 1 (the first dimension) for the axis to be compared
        kwargs: keyword arguments for numpy.allclose()

    Returns:
        A (1+n)D numpy array with unique rows.
    """
    work_arr = arr.copy()
    idx = 0
    while idx < work_arr.shape[0]:
        this_arr = work_arr[idx]
        if_not_close = (idx + 1) * [True] + [
            not numpy.allclose(this_arr, work_arr[i], **kwargs) for i in range(idx + 1, work_arr.shape[0])
        ]
        work_arr = work_arr[if_not_close]
        idx += 1
    return work_arr


def get_twist_property(m: int, r: int):
    """
    Equations from: (please check the appendix)
    J. M. B. Lopes dos Santos, N. M. R. Peres, and A. H. Castro Neto, Continuum model of the twisted graphene bilayer, Phys. Rev. B 86, 155449 (2012).
    """
    # check coprime
    if numpy.gcd(m, r) != 1:
        raise ValueError(f"m={m} and r={r} are not coprime")
        # return 0.0
    numerator = 3 * m**2 + 3 * m * r + r**2 / 2
    denominator = 3 * m**2 + 3 * m * r + r**2
    angle = numpy.arccos(numerator / denominator)
    if numpy.gcd(r, 3) == 1:
        suplat = numpy.array(
            [
                [m, m + r],
                [-m - r, 2 * m + r],
            ]
        )
    elif numpy.gcd(r, 3) == 3:
        suplat = numpy.array(
            [
                [m + r / 3, r / 3],
                [-r / 3, m + 2 * r / 3],
            ]
        )
    else:
        raise ValueError(f"gcd(r,3) should be 1 or 3, but got {numpy.gcd(r,3)}")
    return {
        "angle": angle,  # fix one lattice, rotate the other
        "suplat_trans": suplat,  # the superlattice follows this transformation
    }


def get_twist_property_3d(*args, **kwargs):
    ret = get_twist_property(*args, **kwargs)
    trans_mat = numpy.diag([1, 1, 1])
    trans_mat[:2, :2] = ret["suplat_trans"]
    ret["suplat_trans"] = trans_mat
    return ret


class ExtendHexagon2D:
    """extend the hexagon lattice to a superlattice"""

    hexagon_lat_vec = numpy.array(
        [  # each row is a lattice vector
            [
                1.0,
                0.0,
            ],
            [
                0.5,
                3**0.5 / 2,
            ],
            # [0.5, 3**0.5 / 2,],
            # [-0.5, 3**0.5 / 2,],
        ]
    )

    def __init__(self, n1: int, n2: int, rot: float = 0.0, cen1: float = 0, cen2: float = 0):
        """Generate a superlattice and rotate it

        Args:
            n1: int, number of the first lattice vector
            n2: int, number of the second lattice vector
            rot: float, rotation angle in radian
            cen1: float, center of the first lattice vector
            cen2: float, center of the second lattice vector

        Returns:
            an instance of this class

        Examples:
        ```python
        ExtendHexagon2D.from_mr(m, r, multi_n=multi_n)
        ```
        """
        self.n1, self.n2 = n1, n2  # x, y
        self.rot = rot
        self.primlat_vec = numpy.einsum("ij,kj->ki", get_rot2D_mat(rot), self.hexagon_lat_vec)

        self.fulllat_vec = n1 * self.primlat_vec[0] + n2 * self.primlat_vec[1]
        self.fulllat_center = cen1 * self.primlat_vec[0] + cen2 * self.primlat_vec[1]

    def __repr__(self):
        return f"{self.__class__.__name__}(n1={self.n1}, n2={self.n2})"

    @property
    def rotation_angle(self) -> float:
        """Return the rotation angle in radian"""
        return self.rot

    @property
    def sublattice_vectors(self) -> numpy.ndarray:
        """Return the translation vectors of each sublattice in the superlattice
        shape(nth_sublat, xy)
        """
        return numpy.stack(
            [i1 * self.primlat_vec[0] + i2 * self.primlat_vec[1] for i1 in range(self.n1) for i2 in range(self.n2)]
        ) - self.fulllat_center.reshape(1, 2)

    @classmethod
    def from_mr(cls, m: int, r: int, n1: int = None, n2: int = None, multi_n: int = 2, rot: bool = True):
        """generate a superlattice from m and r = n - m (moire rotation is optional)

        Args:
            m: int, the first integer, defines the twist angle
            r: int, the second integer, defines the twist angle
            n1: int, superlattice size in the first lattice vector
            n2: int, superlattice size in the second lattice vector
            multi_n: int, the multiplier of the minimum number
            rot: bool, whether to rotate the superlattice by the twist angle

        Returns:
            an instance of this class
        """
        twist_prop = get_twist_property(m, r)
        angle, suplat_trans = twist_prop["angle"], twist_prop["suplat_trans"]
        min_n = int(numpy.ceil(numpy.max(suplat_trans)))
        if n1 is None or n2 is None:
            n1, n2 = multi_n * min_n, multi_n * min_n
        if rot:
            return cls(n1, n2, angle, 0, n2 // 2)
        else:
            return cls(n1, n2, 0, 0, n2 // 2)

    def viz(self, ax=None, center=False, **kwargs):
        """Visualize the superlattice

        Args:
            ax: matplotlib axis
            center: bool, whether to center the superlattice at (0,0)
            kwargs: keyword arguments for matplotlib.pyplot.plot()

        Returns:
            matplotlib axis
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        _basic_line_1 = numpy.array(
            [
                [
                    0.0,
                    0.0,
                ],
                self.primlat_vec[0],
            ]
        )

        _basic_line_2 = numpy.array(
            [
                [
                    0.0,
                    0.0,
                ],
                self.primlat_vec[1],
            ]
        )

        primlat_lines = numpy.stack(
            [  # shape (nth_line, start_end, xy)
                _basic_line_1,
                _basic_line_2,
                _basic_line_1 + self.primlat_vec[1],
                _basic_line_2 + self.primlat_vec[0],
            ]
        )

        concate_alllat_lines = numpy.concatenate(
            [  # shape (nth_line, start_end, xy)
                primlat_lines + (i1 * self.primlat_vec[0] + i2 * self.primlat_vec[1]).reshape(1, 1, 2)
                for i1 in range(self.n1)
                for i2 in range(self.n2)
            ]
        )

        # remove duplicate lines
        alllat_lines = get_unique_arr(concate_alllat_lines)

        if center:
            alllat_lines -= self.fulllat_center.reshape(1, 1, 2)

        if "color" not in kwargs:
            kwargs["color"] = "black"
        ax.plot(*numpy.moveaxis(alllat_lines, [0, 1, 2], [2, 1, 0]), **kwargs)
        ax.set_aspect("equal")
        return ax


class _MoireHexagon2D:
    """Generate a Moire superlattice from m and r = n - m, without sublattice information

    Examples:
    ```python
    _MoireHexagon2D(m, r).vertices
    _MoireHexagon2D(m, r).moire_angle
    _MoireHexagon2D(m, r).viz(ax=ax, color="green")
    ```
    """

    hexagon_lat_vec = ExtendHexagon2D.hexagon_lat_vec

    def __init__(self, m: int, r: int):
        """
        Args:
            m: int, the first integer, defines the twist angle
            r: int, the second integer, defines the twist angle
            (for more info, please refer to get_twist_property() function)
        """
        self.m, self.r = m, r
        twist_prop = get_twist_property(m, r)
        self.angle, self.suplat_trans = twist_prop["angle"], twist_prop["suplat_trans"]
        _log.info(f"twisting angle: {self.angle:.3f} rad")
        self.suplat_vec = numpy.einsum("ij,jk->ik", self.suplat_trans, self.hexagon_lat_vec)
        self.suplat_vec = numpy.einsum("ij,kj->ki", get_rot2D_mat(-numpy.pi / 3), self.suplat_vec)

    @property
    def moire_angle(self) -> float:
        """Return the moire angle in radian"""
        return self.angle

    @property
    def lattice_vector(self) -> numpy.ndarray:
        """Return the lattice vectors of the superlattice, shape (nth_lattice_vector, xy)"""
        return self.suplat_vec

    @property
    def vertices(self) -> numpy.ndarray:
        """Return the vertices of the superlattice, shape (nth_vertex, xy)"""
        return numpy.stack(
            [
                numpy.array([0.0, 0.0]),
                self.suplat_vec[0],
                self.suplat_vec[0] + self.suplat_vec[1],
                self.suplat_vec[1],
            ]
        )

    def viz(self, ax=None, **kwargs):
        """Visualize the superlattice, edges only"""
        import matplotlib.pyplot as plt

        _basic_line_1 = numpy.array(
            [
                [
                    0.0,
                    0.0,
                ],
                self.suplat_vec[0],
            ]
        )

        _basic_line_2 = numpy.array(
            [
                [
                    0.0,
                    0.0,
                ],
                self.suplat_vec[1],
            ]
        )

        suplat_lines = numpy.stack(
            [  # shape (nth_line, start_end, xy)
                _basic_line_1,
                _basic_line_2,
                _basic_line_1 + self.suplat_vec[1],
                _basic_line_2 + self.suplat_vec[0],
            ]
        )

        if ax is None:
            ax = plt.gca()
        if "color" not in kwargs:
            kwargs["color"] = "black"
        ax.plot(*numpy.moveaxis(suplat_lines, [0, 1, 2], [2, 1, 0]), **kwargs)
        ax.set_aspect("equal")
        return ax


class SuplatHexagon2D:
    """the moire superlattice

    Examples:
    ```python
    suplat = SuplatHexagon2D(2, 1)
    suplat.cal_layers()
    suplat.viz().grid()
    print(suplat.moire_angle)
    ```
    """

    hexagon_lat_vec = ExtendHexagon2D.hexagon_lat_vec

    def __init__(self, m: int, r: int):
        """Generate a superlattice from m and r = n - m, with sublattice information.

        Args:
            m: int, the first integer, defines the twist angle
            r: int, the second integer, defines the twist angle
            (for more info, please refer to get_twist_property() function)
        """
        self.m, self.r = m, r
        twist_prop = get_twist_property(m, r)
        self.angle, self.suplat_trans = twist_prop["angle"], twist_prop["suplat_trans"]
        _log.info(f"twisting angle: {self.angle:.3f} rad")
        self.suplat_vec = numpy.einsum("ij,jk->ik", self.suplat_trans, self.hexagon_lat_vec)
        self.suplat_vec = numpy.einsum("ij,kj->ki", get_rot2D_mat(-numpy.pi / 3), self.suplat_vec)

    @property
    def moire_angle(self) -> float:
        """Return the moire angle in radian"""
        return self.angle

    @property
    def superlattice_vector(self) -> numpy.ndarray:
        """Return the lattice vectors of the superlattice, shape (nth_lattice_vector, xy)"""
        return self.suplat_vec

    @property
    def vertices(self) -> numpy.ndarray:
        """Return the vertices of the superlattice, shape (nth_vertex, xy)"""
        return numpy.stack(
            [
                numpy.array([0.0, 0.0]),
                self.suplat_vec[0],
                self.suplat_vec[0] + self.suplat_vec[1],
                self.suplat_vec[1],
            ]
        )

    @property
    def sublattice_vectors(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Return the sublattice vectors of the superlattice,
        shape (nth_sublattice, xy) and two of them (unrotated, rotated)"""
        return (self.sublat_lower, self.sublat_upper)

    def cal_layers(self, multi_n: int = 4, delta_r=(1e-4, 1e-5)):
        """calculate the layers' info of the superlattice"""
        from shapely import Point, Polygon, within

        delta_r = numpy.array(delta_r)
        sublat_lower = ExtendHexagon2D.from_mr(self.m, self.r, multi_n=multi_n, rot=False).sublattice_vectors
        sublat_upper = ExtendHexagon2D.from_mr(self.m, self.r, multi_n=multi_n, rot=True).sublattice_vectors

        suplat_polygon = Polygon(self.vertices)
        sublat_lower_in = sublat_lower[[within(Point(i + delta_r), suplat_polygon) for i in sublat_lower]]
        sublat_upper_in = sublat_upper[[within(Point(i + delta_r), suplat_polygon) for i in sublat_upper]]

        nsublat_by_area = int(
            numpy.round(numpy.cross(*self.suplat_vec) / numpy.cross(*ExtendHexagon2D.hexagon_lat_vec))
        )
        assert (
            len(sublat_lower_in) == len(sublat_upper_in) == nsublat_by_area
        ), f"sublat_lower_in: {len(sublat_lower_in)}, sublat_upper_in: {len(sublat_upper_in)}, nsublat_by_area: {nsublat_by_area}"

        self.sublat_lower = sublat_lower_in
        self.sublat_upper = sublat_upper_in

    def viz(self, ax=None, **kwargs):
        """visualize the superlattice

        Args:
            ax: matplotlib axis
            kwargs: keyword arguments for matplotlib.pyplot.plot()

        Returns:
            matplotlib axis
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        _basic_line_1 = numpy.array(
            [
                [
                    0.0,
                    0.0,
                ],
                self.suplat_vec[0],
            ]
        )

        _basic_line_2 = numpy.array(
            [
                [
                    0.0,
                    0.0,
                ],
                self.suplat_vec[1],
            ]
        )

        suplat_lines = numpy.stack(
            [  # shape (nth_line, start_end, xy)
                _basic_line_1,
                _basic_line_2,
                _basic_line_1 + self.suplat_vec[1],
                _basic_line_2 + self.suplat_vec[0],
            ]
        )

        if "color" not in kwargs:
            kwargs["color"] = "black"
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.5
        ax.plot(*numpy.moveaxis(suplat_lines, [0, 1, 2], [2, 1, 0]), **kwargs, zorder=2.1)

        ax.scatter(*self.sublat_lower.T, label="layer 1", s=4, zorder=2.2)
        ax.scatter(*self.sublat_upper.T, label="layer 2", s=2, zorder=2.3)
        ax.set_title(f"Moire angle: {self.moire_angle:.3f} rad = {self.moire_angle * 180 / numpy.pi:.3f} deg")

        ax.set_aspect("equal")
        ax.legend()
        return ax


if __name__ == "__main__":
    pass
