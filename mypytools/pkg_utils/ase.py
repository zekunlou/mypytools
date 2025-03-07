import os
from typing import Union

import numpy
from ase.atoms import Atoms


def match_two_atoms(
    a: Atoms,
    b: Atoms,
    spatial_tolerance: float = 1e-2,
):
    ret_dict = {
        "atoms_indices_a2b": None,
        "atoms_indices_b2a": None,
        "fail_reason": None,
    }
    st = spatial_tolerance
    # check number of atoms
    if len(a) != len(b):
        ret_dict["fail_reason"] = f"len(a)={len(a)} != len(b)={len(b)}"
        return ret_dict
    # check number of atoms by species
    if sorted(a.get_chemical_symbols()) != sorted(b.get_chemical_symbols()):
        ret_dict["fail_reason"] = f"chemical_symbols mismatch"
        return ret_dict
    # check cell
    if not numpy.allclose(a.cell, b.cell, atol=st):
        ret_dict["fail_reason"] = f"cell mismatch"
        return ret_dict
    # check positions, maybe have to permute the atoms
    a = a.copy()
    a.wrap()
    b = b.copy()
    b.wrap()
    atoms_dist = numpy.linalg.norm(a.positions[:, None, :] - b.positions[None, :, :], axis=2)  # shape (natoms, natoms)
    if numpy.sum(atoms_dist < st) != len(a):
        ret_dict["fail_reason"] = f"atoms positions mismatch, too few/many atoms' pairs with distance < {st}"
        return ret_dict
    # find the permutation as an array of indices
    ret_dict["atoms_indices_a2b"] = numpy.argmin(atoms_dist, axis=0)  # b = a[atoms_indices_a2b]
    ret_dict["atoms_indices_b2a"] = numpy.argmin(atoms_dist, axis=1)  # a = b[atoms_indices_b2a]

    return ret_dict


def geom_aims_insert_line(fpath: str, insert_lines: Union[str, list[str]]):
    """Insert a line like `set_vacuum_level` to the geometry.in file of FHI-aims"""

    # identify the last line starting with #, identify the first line starting with `atom` or `lattice_vector`
    assert os.path.exists(fpath)
    with open(fpath) as f:
        lines = f.readlines()
    last_comment_line = max([i for i, line in enumerate(lines) if line.startswith("#")])
    first_geom_line = min(
        [i for i, line in enumerate(lines) if line.startswith("atom") or line.startswith("lattice_vector")]
    )
    assert last_comment_line < first_geom_line, (
        f"{last_comment_line=}, {first_geom_line=}, please check the file {fpath}"
    )
    if isinstance(insert_lines, str):
        insert_lines = [insert_lines]
    lines = lines[:first_geom_line] + insert_lines + lines[first_geom_line:]
    with open(fpath, "w") as f:
        f.writelines(lines)
