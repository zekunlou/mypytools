from typing import Literal, Union

import numpy
from ase.atoms import Atoms
from ase.build import make_supercell


def gen_sliding_bilayer(
    cell_prim: Atoms,
    n1: int,
    n2: int,
    z_dist: float,
    conj: Union[Literal["AA"], Literal["AB"]],
    shake_upper_layer_z: Union[None, float] = None,
    slide_lower_layer_xy: Union[None, float] = None,
    shake_all_atoms: Union[None, float] = None,
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
    """
    if seed is not None:
        numpy.random.seed(seed)
    """make supercell"""
    cell_sup = make_supercell(cell_prim, [[n1, 0, 0], [0, n2, 0], [0, 0, 1]])
    cell_sup.cell[2, 2] = 6 * z_dist
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
    """deal with conj"""
    if conj == "AB":  # invert upper layer
        layer_upper_z_mean = numpy.mean(layer_upper.positions[:, 2])
        layer_upper.positions[:, 2] = 2 * layer_upper_z_mean - layer_upper.positions[:, 2]
    elif conj == "AA":  # do nothing
        pass
    else:
        raise ValueError(f"conj {conj} not supported")
    """shake upper layer z coordinate, only upper layer"""
    if shake_upper_layer_z is not None:
        assert isinstance(shake_upper_layer_z, float)
        shake_val = shake_upper_layer_z * numpy.random.randn()
        layer_upper.positions[:, 2] += shake_val
    """join two layers"""
    cell_comb = Atoms(
        symbols=layer_lower.get_chemical_symbols() + layer_upper.get_chemical_symbols(),
        positions=numpy.concatenate([layer_lower.positions, layer_upper.positions], axis=0),
        cell=cell_sup.cell,
        pbc=cell_sup.pbc,
    )
    """shape atoms"""
    if shake_all_atoms is not None:
        assert isinstance(shake_all_atoms, float)
        M = cell_comb.get_masses()
        shake_val = shake_all_atoms * numpy.random.randn(*cell_comb.positions.shape) * (M ** (-0.5)).reshape(-1, 1)
        cell_comb.positions += shake_val
    return cell_comb


if __name__ == "__main__":
    pass


"""
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
visualize_cell_2d(bilayer_cell);  plt.show()
plt.scatter(bilayer_cell.positions[:, 0], bilayer_cell.positions[:, 2]);  plt.show()
"""
