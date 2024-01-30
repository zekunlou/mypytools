"""
Visualizing cell data (ase.Atoms)
"""

from typing import List

import matplotlib.pyplot as plt
import numpy

# from ase.io import read
from ase.atoms import Atoms


def visualize_cell_2d(
    geom: Atoms,
    alpha_min: float = 0.2,
    alpha_max: float = 1.0,
    base_size: float = 1e1,
    ax=None,
):
    """
    Visualize 2D projection of cell and atoms in the cell (z coordinate as opacity)
    """
    if ax is None:
        ax = plt.gca()

    symbs: List[str] = geom.get_chemical_symbols()
    natoms = len(symbs)
    atomz_arr: numpy.ndarray = geom.get_atomic_numbers()  # Z number of atoms
    coords: numpy.ndarray = geom.get_positions()  # shape (natoms, xyz)
    cell: numpy.ndarray = geom.get_cell()  # shape (lat_vec, xyz)
    zmax, zmin = coords[:, 2].max(), coords[:, 2].min()

    def opacity(z: float):
        """larger z, larger opacity"""
        return alpha_min + (alpha_max - alpha_min) * (z - zmin) / (zmax - zmin)

    z_order = 2.1 + 0.1 * (coords[:, 2] - zmin) / (zmax - zmin)  # from 2.1 to 2.2
    alpha_arr = alpha_min + (alpha_max - alpha_min) * (coords[:, 2] - zmin) / (zmax - zmin)
    alpha_arr[alpha_arr > 1.0] = 1.0
    alpha_arr[alpha_arr < 0.0] = 0.0
    size_arr = base_size / natoms**0.5 * atomz_arr**0.5
    color_dict = {symb: f"C{idx}" for idx, symb in enumerate(set(symbs))}
    color_list = [color_dict[symb] for symb in symbs]

    # plot atoms
    for i in range(natoms):
        ax.plot(
            coords[i, 0],
            coords[i, 1],
            marker="o",
            ms=size_arr[i],
            color=color_list[i],
            alpha=alpha_arr[i],
            zorder=z_order[i],
        )

    # plot bounding box
    cell_2d = [cell[0, :2], cell[1, :2]]
    ax.plot([0, cell_2d[0][0]], [0, cell_2d[0][1]], "k--")
    ax.plot([0, cell_2d[1][0]], [0, cell_2d[1][1]], "k--")
    ax.plot([cell_2d[0][0], cell_2d[0][0] + cell_2d[1][0]], [cell_2d[0][1], cell_2d[0][1] + cell_2d[1][1]], "k--")
    ax.plot([cell_2d[1][0], cell_2d[0][0] + cell_2d[1][0]], [cell_2d[1][1], cell_2d[0][1] + cell_2d[1][1]], "k--")

    ax.set_aspect("equal")
    ax.grid(True)

    return ax


if __name__ == "__main__":
    pass
